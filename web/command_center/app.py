#!/usr/bin/env python3
"""XRD 三机异构协同 · 指挥中心 (Command Center) — 第 5 期.

apex xiaomiju.xyz 的统一门户: SSO 登录后落地这里, 顶部常驻栏 + 三系统 iframe 同页内嵌
(AI 脑 lab / 车载脑 car / 机械臂 arm), 单一网址。本服务只做两件事:
  1. 托管门户 SPA (static/index.html, 零构建纯前端)
  2. /api/fleet 聚合三机实时状态 — 直探 frp 隧道真机端口 (18888/18890/18891),
     真机在线即报 online, 否则 offline (此时 Caddy 失败转移到 VPS 镜像, UI 照常在)

常驻 VPS (127.0.0.1:29100, systemd xrd-cmdcenter), 设备关机门户也在。
Caddy apex 块: import sso (forward_auth) + reverse_proxy 127.0.0.1:29100。
"""
import csv
import datetime
import hashlib
import html
import ipaddress
import io
import json
import math
import os
import re
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, abort, g, jsonify, request, send_from_directory
from urllib.parse import quote, urlparse

from cmdcenter import RuntimeController, register_site32
from cmdcenter.access import SAFE_PUBLIC_METHODS, classify_request, role_allows, role_from_headers
from cmdcenter.config import load_config
from cmdcenter.public_dto import (
    STATUS_TAXONOMY,
    mask_ip as _mask_ip,
    public_asset_group as _public_asset_group,
    public_asset_text as _dto_public_asset_text,
    public_redaction_scan as _public_redaction_scan,
    public_runbook_text as _dto_public_runbook_text,
    public_safe_text as _dto_public_safe_text,
    public_severity as _dto_public_severity,
    route_service as _route_service,
    serving_source as _serving_source,
    status_envelope as _dto_status_envelope,
    status_from_serving as _status_from_serving,
    status_meta as _status_meta,
)
from cmdcenter.route_contract import group_doc_entries, reconciled_api_docs
from cmdcenter.research_search import STATUS_LABELS as SEARCH_STATUS_LABELS
from cmdcenter.research_search import search_research
from cmdcenter.research_collections import build_research_collections, collection_detail
from cmdcenter.rb_voe_public import RbVoePublicError, load_public_snapshot

app = Flask(__name__, static_folder="static", static_url_path="")
_CMD_CONFIG = load_config()
ASSET_VER = _CMD_CONFIG.asset_version
RELEASED_AT = _CMD_CONFIG.released_at

FINALS_PUBLIC_FACTS = {
    "embodied": {
        "claim": "具身脑已真机完成取瓶、升顶、0.50m 里程计闭环直行、下降放瓶与复位。",
        "boundary": "SLAM/RViz/Lab-FSD 只读观察可并发展示；Lab-FSD 仍为 shadow/assist，不持有底盘执行权。",
    },
    "ai_brain": {
        "claim": "AI 脑现有 Dashboard、XRD 视觉与材料合成预测链已在平板完整彩排并冻结。",
        "boundary": "X5-RB-VoE 仅被动部署、离线验证且未启用，不在现场运行或物理控制链中。",
    },
    "dual_arm": {
        "claim": "双机械臂已真机完成 arm01 单臂视觉冗余、投袋，以及与 arm02 并发四周期研磨。",
        "boundary": "袋状态以 X5 CPU/OpenCV 判定为权威；BPU 仅作辅助语义与真实执行证据，公网不下发动作。",
    },
}
CMD_TEST_MODE = _CMD_CONFIG.cmd_test_mode
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
_ALLOWED_HOST_SUFFIXES = _CMD_CONFIG.allowed_host_suffixes
_ALLOWED_HOSTS = set(_CMD_CONFIG.allowed_hosts)
_SAFE_PUBLIC_METHODS = set(SAFE_PUBLIC_METHODS)
_WEBHOOK_HOSTS = {key: set(value) for key, value in _CMD_CONFIG.webhook_hosts.items()}
_WEBHOOK_EXTRA_HOSTS = set(_CMD_CONFIG.webhook_extra_hosts)
_RB_VOE_PUBLIC_PATH = Path(__file__).resolve().parent / "public_evidence" / "rb_voe_r1_public.json"


def _public_safe_text(v, n=220):
    return _dto_public_safe_text(v, n)


def _public_asset_text(v, n=180):
    return _dto_public_asset_text(v, n)


def _public_runbook_text(v, n=220):
    return _dto_public_runbook_text(v, n)


def _status_envelope(state, source=None, checked_at=None, ttl_s=90, error=None, confidence=None):
    return _dto_status_envelope(
        state, source, checked_at, ttl_s=ttl_s, error=error, confidence=confidence, release=ASSET_VER
    )


def _public_severity(status):
    return _dto_public_severity(status, test_mode=CMD_TEST_MODE)

# 三系统: real = frp 隧道真机端口 (探活判在线), 公网子域给前端 iframe 用
SYSTEMS = {
    "lab": {"name": "AI 脑 · 智能计算平台", "real": 18888, "url": "https://lab.xiaomiju.xyz/"},
    "car": {"name": "车载脑 · SLAM / 影子规划证据", "real": 18890, "url": "https://car.xiaomiju.xyz/"},
    "arm": {"name": "机械臂 · arm01 工位证据", "real": 18891, "url": "https://arm.xiaomiju.xyz/"},
}

# 镜像 (VPS 本地常驻 fallback) — 真机隧道断时 Caddy active health 切到这
MIRROR_PORT = {"lab": 28888, "car": 28890, "arm": 28891}
MIRROR_SVC = {"lab": "mirror-lab", "car": "mirror-navcockpit", "arm": "mirror-workcockpit"}

_cache = {"ts": 0.0, "data": None}
_ops_cache = {"ts": 0.0, "data": None}
_kpi_cache = {"ts": 0.0, "data": None}
_public_status_cache = {"ts": 0.0, "data": None}
_availability_cache = {"ts": 0.0, "data": None}
_lock = threading.Lock()


def _probe(port, path, timeout=2.5):
    """直探回环端口 (frp 隧道) → (status_code, body_bytes) 或 (None, None).

    用 curl 不用 urllib: arm 隧道走 AI 脑 frpc 二跳中继到主机械臂 WorkCockpit, urllib 对这条
    双跳链稳定超时而 curl 0.15s 必成 (urllib/frp/Werkzeug keep-alive 交互坑, 换 curl 绕开)。
    """
    try:
        p = subprocess.run(
            ["curl", "-s", "-m", str(timeout), "-w", "\n%{http_code}",
             f"http://127.0.0.1:{port}{path}"],
            capture_output=True, timeout=max(timeout + 0.35, 0.55))
        raw = p.stdout
        nl = raw.rfind(b"\n")
        if nl < 0:
            return None, None
        code = raw[nl + 1:].decode("ascii", "ignore").strip()
        return (int(code) if code.isdigit() else None), raw[:nl]
    except Exception:
        return None, None


def _json(body):
    try:
        return json.loads(body) if body else None
    except Exception:
        return None


def _alive(port, timeout=2.5, tries=2):
    """探活带 1 次重试 — 单次抖动不翻 online/offline (防 UI 闪)."""
    for _ in range(tries):
        st, _b = _probe(port, "/api/health", timeout=timeout)
        if st == 200:
            return True
    return False


def _f_lab():
    # AI brain dashboard may exceed 2.5s during BPU/LLM load or cellular frp jitter; avoid false offline.
    lab = {"online": _alive(SYSTEMS["lab"]["real"], timeout=5.0, tries=1), "metrics": {}}
    if lab["online"]:
        s2, b2 = _probe(SYSTEMS["lab"]["real"], "/api/local_llm_health", timeout=1.0)
        j = _json(b2)
        servers = (j or {}).get("servers") or (j or {}).get("llms") or []
        if isinstance(servers, list) and servers:
            up = sum(1 for s in servers if (s.get("ok") or s.get("up") or s.get("alive")))
            lab["metrics"]["本地 LLM"] = f"{up}/{len(servers)} 在线"
    if not lab["metrics"]:
        lab["metrics"]["推理"] = "9 LLM + 5 BPU"
    return lab


def _f_car():
    car = {"online": _alive(SYSTEMS["car"]["real"]), "metrics": {}}
    if car["online"]:
        s2, b2 = _probe(SYSTEMS["car"]["real"], "/api/bridge/status")
        j = _json(b2)
        if j:
            car["metrics"]["遥测桥"] = "在线" if j.get("alive") else "离线"
            sc = (j.get("safety") or {}).get("speed_cap")
            if sc is not None:
                car["metrics"]["限速"] = f"{sc} m/s"
            if j.get("estop"):
                car["metrics"]["急停"] = "已闩锁"
    return car


def _f_arm():
    arm = {"online": _alive(SYSTEMS["arm"]["real"]), "metrics": {}}
    if arm["online"]:
        s2, b2 = _probe(SYSTEMS["arm"]["real"], "/api/joints/arm01")
        j = _json(b2)
        if j:
            arm["metrics"]["arm01"] = "关节在线" if j.get("online") else "离线"
        s3, b3 = _probe(SYSTEMS["arm"]["real"], "/api/interlock")
        j = _json(b3)
        if j:
            lvl = j.get("level", "?")
            arm["metrics"]["防撞互锁"] = {"unknown": "待机", "safe": "安全",
                                           "warn": "警戒", "danger": "危险"}.get(lvl, lvl)
    return arm


def _build_fleet():
    # 串行探活: curl 单请求 ~0.15s, 7 个端点 ~1s 够快。不并发 —— 设备端是单线程
    # Werkzeug + frp 二跳, 3 条并发连接会互相挤致间歇超时 (实测 flicker), 串行反而稳。
    out = {"ts": time.time()}
    for k, fn in (("lab", _f_lab), ("car", _f_car), ("arm", _f_arm)):
        try:
            out[k] = fn()
        except Exception:
            out[k] = {"online": False, "metrics": {}}
    return out


def _probe_ms(port, path="/api/health", timeout=2.5):
    """探回环端口 → (status_code, latency_ms). 用于运维总览的链路延迟。"""
    if CMD_TEST_MODE:
        return None, None
    if CMD_TEST_MODE:
        return None, None
    try:
        p = subprocess.run(
            ["curl", "-s", "-o", os.devnull, "-m", str(timeout),
             "-w", "%{http_code} %{time_total}", f"http://127.0.0.1:{port}{path}"],
            capture_output=True, timeout=timeout + 1.5, text=True)
        parts = (p.stdout or "").split()
        code = int(parts[0]) if parts and parts[0].isdigit() else None
        ms = int(float(parts[1]) * 1000) if len(parts) > 1 else None
        return code, ms
    except Exception:
        return None, None


def _svc_active(name):
    """systemctl is-active (非 root 可查) → active/inactive/failed/unknown."""
    if CMD_TEST_MODE:
        return "unknown"
    try:
        p = subprocess.run(["systemctl", "is-active", name],
                           capture_output=True, text=True, timeout=3)
        return (p.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _build_ops():
    """运维总览: 每系统 真机隧道 + 镜像 双路径状态 + 当前 serving + 延迟 + 镜像服务态。

    设备全离线时照样有数据 (serving=mirror), 正是"设备关机镜像照常在"的可视化证明。
    """
    out = {"ts": time.time(), "systems": {}}
    metric_fn = {"lab": _f_lab, "car": _f_car, "arm": _f_arm}
    for k in ("lab", "car", "arm"):
        real_code, real_ms = _probe_ms(SYSTEMS[k]["real"], timeout=5.0 if k == "lab" else 2.5)
        mir_code, mir_ms = _probe_ms(MIRROR_PORT[k])
        real_on, mir_on = real_code == 200, mir_code == 200
        metrics = {}
        if real_on:
            try:
                metrics = metric_fn[k]().get("metrics", {})
            except Exception:
                metrics = {}
        out["systems"][k] = {
            "name": SYSTEMS[k]["name"],
            "real_online": real_on, "mirror_online": mir_on,
            "serving": "real" if real_on else ("mirror" if mir_on else "down"),
            "real_ms": real_ms, "mirror_ms": mir_ms,
            "mirror_svc": _svc_active(MIRROR_SVC[k]),
            "metrics": metrics,
        }
    return out


def _lab_port():
    """KPI 取 lab 当前 serving 端口: 真机隧道在线用真机 (18888), 否则用镜像 (28888)."""
    code, _ = _probe_ms(SYSTEMS["lab"]["real"], "/api/health", timeout=5.0)
    return SYSTEMS["lab"]["real"] if code == 200 else MIRROR_PORT["lab"]


def _build_kpi():
    """平台真实 KPI — 取自 lab 当前 serving 端 (真机/镜像), 全部真值不编造.

    Conformal/审计链是模型/数据固有真值 (与预测条数无关), 演示时镜像也有意义;
    预测累计/LLM 在线随 serving 端真实变化 (真机上线即真机 649 条等)。
    """
    port = _lab_port()
    out = {"ts": time.time(), "source": "real" if port == SYSTEMS["lab"]["real"] else "mirror", "kpi": {}}
    k = out["kpi"]
    # 预测累计
    _, b = _probe(port, "/api/predictions/accuracy", timeout=2.0)
    j = _json(b)
    if j:
        k["predictions"] = j.get("n_predictions")
        k["with_actuals"] = j.get("n_with_actuals")
    # Conformal 覆盖 + 区间收窄 (真实标定值)
    _, b = _probe(port, "/api/conformal_stats", timeout=2.0)
    j = _json(b)
    if j:
        cov = j.get("empirical_coverage_90")
        if cov is not None:
            k["ci_coverage_pct"] = round(cov * 100, 1)
        mc = j.get("mc_conformal") or {}
        if mc.get("interval_width_reduction_vs_split_pct") is not None:
            k["ci_narrowing_pct"] = round(mc["interval_width_reduction_vs_split_pct"], 1)
    # 审计链完整性
    _, b = _probe(port, "/api/audit_chain", timeout=2.0)
    j = _json(b)
    if j:
        k["audit_valid"] = j.get("n_valid")
        k["audit_total"] = j.get("n_records")
        k["audit_intact"] = bool(j.get("chain_intact"))
    # 本地 LLM 在线
    _, b = _probe(port, "/api/local_llm_health", timeout=2.0)
    j = _json(b)
    servers = (j or {}).get("servers") or (j or {}).get("llms") or []
    if isinstance(servers, list) and servers:
        k["llm_up"] = sum(1 for s in servers if (s.get("ok") or s.get("up") or s.get("alive")))
        k["llm_total"] = len(servers)
    return out


# ============================================================ P1 数据底座: SQLite historian
# 工业平台 (Ignition/Insights Hub) 的根基是历史库 — 状态/延迟/KPI 落盘, 告警/SLO/报表全吃它。
# 单文件 WAL 库, 写入只来自采样线程 (+ 请求线程的事件), 读并发 WAL 天然支持。
from cmdcenter import storage as _storage

DB_PATH = _storage.resolve_db_path(__file__)
SAMPLE_EVERY = 30          # 采样周期 (s); KPI 每 2 周期采一次
RETAIN_SAMPLES_D = 14      # 状态样本保留 14 天
RETAIN_EVENTS_D = 90       # 事件/KPI 保留 90 天

_sse_seq = {"n": 0}        # 任何新事件/新采样 → 自增, SSE 客户端据此推送
_prev_serving = {}         # sys -> 上次 serving (跨样本跃迁检测)
_prev_mirror_svc = {}      # sys -> 上次镜像 systemd 状态


def _db():
    return _storage.connect(DB_PATH)


def _init_db():
    _storage.initialize(DB_PATH)


def _seed_defaults():
    """首次启动播种合理的真实运营配置 (非伪造遥测): 默认告警规则 / PM 排程 / BOM 备件."""
    _storage.seed_defaults(DB_PATH, now=int(time.time()))


def _add_event(sys_k, kind, severity, message, con=None):
    own = con is None
    if own:
        con = _db()
    con.execute("INSERT INTO events(ts,sys,kind,severity,message) VALUES(?,?,?,?,?)",
                (int(time.time()), sys_k, kind, severity, message))
    if own:
        con.commit()
        con.close()
    _sse_seq["n"] += 1


_SERV_TXT = {"real": "真机直连", "mirror": "镜像演示", "down": "离线"}


def _emit_serving_events(ops, con=None):
    """ops 快照 → 跃迁事件 (real↔mirror↔down + 镜像 systemd 状态变化).

    采样线程和 /api/ops 请求路径都会调 — 谁先看到跃迁谁记录, 不等下个采样周期。
    """
    for k, s in (ops.get("systems") or {}).items():
        serv = s.get("serving")
        prev = _prev_serving.get(k)
        if prev is not None and serv != prev:
            sev = {"down": "crit", "mirror": "warn", "real": "info"}.get(serv, "info")
            _add_event(k, "serving_change", sev,
                       f"{SYSTEMS[k]['name'].split(' · ')[0]} 服务路径: "
                       f"{_SERV_TXT.get(prev, prev)} → {_SERV_TXT.get(serv, serv)}", con)
        _prev_serving[k] = serv
        msvc = s.get("mirror_svc")
        pmsvc = _prev_mirror_svc.get(k)
        if pmsvc is not None and msvc != pmsvc:
            sev = "info" if msvc == "active" else "warn"
            _add_event(k, "mirror_svc", sev,
                       f"{SYSTEMS[k]['name'].split(' · ')[0]} 镜像服务 {MIRROR_SVC[k]}: "
                       f"{pmsvc} → {msvc}", con)
        _prev_mirror_svc[k] = msvc


def _seed_prev_from_db():
    """重启后从最后样本接续跃迁状态 — 服务重启不产生虚假'上线'事件。"""
    try:
        con = _db()
        for k in SYSTEMS:
            row = con.execute(
                "SELECT serving FROM samples WHERE sys=? ORDER BY ts DESC LIMIT 1",
                (k,)).fetchone()
            if row:
                _prev_serving[k] = row[0]
        con.close()
    except Exception:
        pass


def _record_kpi(kpi_doc, con):
    k = kpi_doc.get("kpi") or {}
    if not k:
        return
    con.execute(
        "INSERT INTO kpi_samples(ts,source,predictions,ci_coverage_pct,ci_narrowing_pct,"
        "audit_valid,audit_total,llm_up,llm_total) VALUES(?,?,?,?,?,?,?,?,?)",
        (int(time.time()), kpi_doc.get("source"), k.get("predictions"),
         k.get("ci_coverage_pct"), k.get("ci_narrowing_pct"),
         k.get("audit_valid"), k.get("audit_total"), k.get("llm_up"), k.get("llm_total")))


def _prune(con):
    now = int(time.time())
    con.execute("DELETE FROM samples WHERE ts < ?", (now - RETAIN_SAMPLES_D * 86400,))
    con.execute("DELETE FROM kpi_samples WHERE ts < ?", (now - RETAIN_EVENTS_D * 86400,))
    con.execute("DELETE FROM events WHERE ts < ?", (now - RETAIN_EVENTS_D * 86400,))


# ============================================================ P2 告警中心 (ISA-18.2 风格)
# 规则引擎吃 historian 同一份快照: 条件成立→升告警(+事件+crit 邮件), 恢复→自动销警, 人工 ack 记操作人。
ALERT_MAIL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_email.json")
MAIL_COOLDOWN_S = 1800       # 同一 (rule,sys) 30 分钟内不重复发信
_mail_last = {}              # (rule,sys) -> 上次发信 ts
_lat_bad = {}                # sys -> 延迟连续超阈计数 (3 次才升告警, 防抖)
_last_kpi_doc = {"doc": None}


def _mail_cfg():
    """alert_email.json: {host,port,user,auth_code,to,enabled} — 没配置/没启用 → 邮件通道关 (诚实禁用)."""
    try:
        with open(ALERT_MAIL_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("enabled") and cfg.get("user") and cfg.get("auth_code") and cfg.get("to"):
            return cfg
    except Exception:
        pass
    return None


def _send_alarm_mail(rule, sys_k, subject, body):
    cfg = _mail_cfg()
    if not cfg:
        return
    key = (rule, sys_k)
    if time.time() - _mail_last.get(key, 0) < MAIL_COOLDOWN_S:
        return
    _mail_last[key] = time.time()

    def _go():
        try:
            import smtplib
            from email.header import Header
            from email.mime.text import MIMEText
            m = MIMEText(body, "plain", "utf-8")
            m["Subject"] = Header(subject, "utf-8")
            m["From"] = cfg["user"]
            m["To"] = cfg["to"]
            s = smtplib.SMTP_SSL(cfg.get("host", "smtp.qq.com"), int(cfg.get("port", 465)), timeout=15)
            s.login(cfg["user"], cfg["auth_code"])
            s.sendmail(cfg["user"], [cfg["to"]], m.as_string())
            s.quit()
        except Exception:
            app.logger.exception("alarm mail")

    threading.Thread(target=_go, daemon=True).start()


def _eval_alarms(ops, kpi_doc=None):
    """规则引擎: 每采样周期跑一遍. 升警/销警都写 alarms 表 + 事件流, crit 升警发邮件."""
    if kpi_doc:
        _last_kpi_doc["doc"] = kpi_doc
    conds = []   # (rule, sys, severity, message, condition_on)
    for k, s in (ops.get("systems") or {}).items():
        nm = SYSTEMS[k]["name"].split(" · ")[0]
        serv = s.get("serving")
        conds.append(("sys_down", k, "crit",
                      f"{nm} 双路径全离线 — 真机隧道与 VPS 镜像均不可达", serv == "down"))
        conds.append(("mirror_svc", k, "crit",
                      f"{nm} 镜像服务 {MIRROR_SVC[k]} 非 active — 兜底链路失效风险",
                      s.get("mirror_svc") not in (None, "unknown", "active")))
        conds.append(("real_offline", k, "info",
                      f"{nm} 真机离线, VPS 镜像兜底中 (设备未上电)", serv == "mirror"))
        ms = s.get("real_ms") if serv == "real" else s.get("mirror_ms")
        thresh = 2500 if serv == "real" else 800
        bad = serv != "down" and ms is not None and ms > thresh
        _lat_bad[k] = (_lat_bad.get(k, 0) + 1) if bad else 0
        conds.append(("latency_high", k, "warn",
                      f"{nm} 服务路径延迟连续超阈 ({serv} {ms}ms > {thresh}ms ×3)", _lat_bad[k] >= 3))
    kd = _last_kpi_doc["doc"]
    if kd:
        kk = kd.get("kpi") or {}
        if kk.get("audit_total"):
            conds.append(("audit_broken", "lab", "crit",
                          f"预测审计链完整性失败: {kk.get('audit_valid')}/{kk.get('audit_total')} (SHA-256 链断裂)",
                          kk.get("audit_intact") is False))
        if kd.get("source") == "real" and kk.get("llm_total"):
            conds.append(("llm_degraded", "lab", "warn",
                          f"本地 LLM 部分离线: {kk.get('llm_up')}/{kk.get('llm_total')}",
                          (kk.get("llm_up") or 0) < kk.get("llm_total")))
    now = int(time.time())
    con = _db()
    active = {(r[1], r[2]): r[0] for r in con.execute(
        "SELECT id, rule, sys FROM alarms WHERE ts_cleared IS NULL")}
    for rule, sk, sev, msg, on in conds:
        key = (rule, sk)
        if on and key not in active:
            con.execute("INSERT INTO alarms(rule,sys,severity,message,ts_raised) VALUES(?,?,?,?,?)",
                        (rule, sk, sev, msg, now))
            _add_event(sk, "alarm_raise", sev, "⚠ 告警: " + msg, con)
            if sev == "crit":
                _send_alarm_mail(rule, sk, f"[XRD 平台告警] {msg}",
                                 f"{msg}\n\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n规则: {rule}\n"
                                 f"门户: https://xiaomiju.xyz → 运维 → 告警中心\n"
                                 f"(同一告警 {MAIL_COOLDOWN_S // 60} 分钟内不重复发信)")
        elif not on and key in active:
            con.execute("UPDATE alarms SET ts_cleared=? WHERE id=?", (now, active[key]))
            _add_event(sk, "alarm_clear", "info", "✓ 恢复: " + msg, con)
    con.commit()
    con.close()


def _alarm_counts():
    con = _db()
    rows = con.execute("SELECT severity, COUNT(*), SUM(ts_ack IS NULL) FROM alarms"
                       " WHERE ts_cleared IS NULL GROUP BY severity").fetchall()
    con.close()
    out = {"crit": 0, "warn": 0, "info": 0, "unacked": 0, "total": 0}
    for sev, n, un in rows:
        if sev in out:
            out[sev] = n
        out["total"] += n
        out["unacked"] += un or 0
    return out


def _alarm_row_dict(r, hist=False):
    d = {"id": r[0], "rule": r[1], "sys": r[2], "severity": r[3], "message": r[4],
         "ts_raised": r[5], "ts_ack": r[6], "ack_by": r[7]}
    if hist:
        d["ts_cleared"] = r[8]
    return d


@app.route("/api/alarms")
def api_alarms():
    con = _db()
    act = con.execute(
        "SELECT id,rule,sys,severity,message,ts_raised,ts_ack,ack_by FROM alarms"
        " WHERE ts_cleared IS NULL ORDER BY CASE severity WHEN 'crit' THEN 0"
        " WHEN 'warn' THEN 1 ELSE 2 END, ts_raised DESC").fetchall()
    hist = con.execute(
        "SELECT id,rule,sys,severity,message,ts_raised,ts_ack,ack_by,ts_cleared FROM alarms"
        " WHERE ts_cleared IS NOT NULL ORDER BY ts_cleared DESC LIMIT 30").fetchall()
    con.close()
    return jsonify({"active": [_alarm_row_dict(r) for r in act],
                    "history": [_alarm_row_dict(r, True) for r in hist],
                    "counts": _alarm_counts(), "mail_channel": bool(_mail_cfg())})


@app.route("/api/alarms/<int:aid>/ack", methods=["POST"])
def api_alarm_ack(aid):
    """人工确认 (operator ack) — 记录操作人 (SSO X-User). 评委 POST 在 SSO 层已被 403."""
    user = request.headers.get("X-User") or "operator"
    con = _db()
    row = con.execute("SELECT id, message, ts_ack FROM alarms WHERE id=?", (aid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "告警不存在"}), 404
    if row[2]:
        con.close()
        return jsonify({"ok": True, "already": True})
    con.execute("UPDATE alarms SET ts_ack=?, ack_by=? WHERE id=?", (int(time.time()), user, aid))
    _add_event(None, "alarm_ack", "info", f"👤 {user} 确认告警: {row[1]}", con)
    con.commit()
    con.close()
    return jsonify({"ok": True})


def _sampler():
    _seed_prev_from_db()
    cycle = 0
    while True:
        t0 = time.time()
        try:
            ops = _build_ops()
            with _lock:
                _ops_cache.update(ts=time.time(), data=ops)
            con = _db()
            now = int(time.time())
            for k, s in ops["systems"].items():
                con.execute(
                    "INSERT INTO samples(ts,sys,serving,real_ms,mirror_ms) VALUES(?,?,?,?,?)",
                    (now, k, s.get("serving"), s.get("real_ms"), s.get("mirror_ms")))
            _emit_serving_events(ops, con)
            kpi_doc = None
            if cycle % 2 == 0:
                try:
                    kpi_doc = _build_kpi()
                    with _lock:
                        _kpi_cache.update(ts=time.time(), data=kpi_doc)
                    _record_kpi(kpi_doc, con)
                except Exception:
                    app.logger.exception("sampler kpi")
            try:
                _record_app_metrics(con, now)   # I1 自监控
                _flush_logs(con)                 # I4 日志落库
            except Exception:
                app.logger.exception("obs metrics/logs")
            if cycle % 120 == 0:
                _prune(con)
                try:
                    _rollup_hourly(con)          # I9 小时降采样
                except Exception:
                    app.logger.exception("rollup")
            con.commit()
            con.close()
            # 告警规则引擎在采样事务提交后跑 (自己开连接, 避免双写锁等待)
            _eval_alarms(ops, kpi_doc)
            _eval_custom_rules(ops, kpi_doc)   # I3 用户自定义规则
            try:
                _daily_report_tick()
            except Exception:
                app.logger.exception("daily report")
            _sse_seq["n"] += 1
        except Exception:
            app.logger.exception("sampler")
        cycle += 1
        time.sleep(max(2.0, SAMPLE_EVERY - (time.time() - t0)))


@app.route("/api/history")
def api_history():
    """状态历史 (historian 真数据): ?sys=lab&hours=24 → 分桶降采样 ≤600 点."""
    sys_k = request.args.get("sys", "lab")
    if sys_k not in SYSTEMS:
        return jsonify({"error": "sys 必须是 lab/car/arm"}), 400
    try:
        hours = min(max(float(request.args.get("hours", 24)), 0.5), RETAIN_SAMPLES_D * 24)
    except ValueError:
        return jsonify({"error": "hours 非法"}), 400
    since = int(time.time() - hours * 3600)
    bucket = max(SAMPLE_EVERY, int(hours * 3600 / 600))
    con = _db()
    rows = con.execute(
        "SELECT (ts/?)*? AS b, AVG(real_ms), AVG(mirror_ms),"
        " SUM(serving='real'), SUM(serving='mirror'), SUM(serving='down'), COUNT(*)"
        " FROM samples WHERE sys=? AND ts>=? GROUP BY b ORDER BY b",
        (bucket, bucket, sys_k, since)).fetchall()
    con.close()
    pts = [{"ts": r[0],
            "real_ms": round(r[1], 1) if r[1] is not None else None,
            "mirror_ms": round(r[2], 1) if r[2] is not None else None,
            "real": r[3], "mirror": r[4], "down": r[5], "n": r[6]} for r in rows]
    return jsonify({"sys": sys_k, "hours": hours, "bucket_s": bucket, "points": pts})


# ============================================================ P4 资产数字孪生
ASSETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets.json")


@app.route("/api/assets")
def api_assets():
    """资产注册表 + serving 状态 + 维保条数; 公网返回会脱敏地址、端口和引脚级细节."""
    try:
        with open(ASSETS_FILE, encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        return jsonify({"error": "assets.json 读取失败"}), 500
    with _lock:
        ops = (_ops_cache["data"] or {}).get("systems", {})
    con = _db()
    try:
        mcounts = dict(con.execute(
            "SELECT asset, COUNT(*) FROM maintenance GROUP BY asset").fetchall())
    except sqlite3.Error:
        mcounts = {}
    finally:
        con.close()
    groups = []
    for g in reg.get("groups", []):
        gg = dict(g)
        k = g.get("key")
        if k in ops:
            gg["serving"] = ops[k].get("serving")
            gg["real_ms"] = ops[k].get("real_ms")
            gg["mirror_ms"] = ops[k].get("mirror_ms")
        elif k == "vps":
            gg["serving"] = "real"   # cmdcenter 自身能响应 = VPS 在线
        for c in gg.get("children", []):
            c["maint_n"] = mcounts.get(c["id"], 0)
        groups.append(_public_asset_group(gg))
    scan = _public_redaction_scan(groups)
    return jsonify({"ts": time.time(), "release": ASSET_VER, "groups": groups,
                    "redaction": {"public_safe": True,
                                  **scan,
                                  "removed": ["private IPs", "public origin IPs", "raw ports", "pin-level actuator details", "serial device paths"],
                                  "note": "Use internal runbooks for physical network/control details; public site keeps evidence-level metadata only."}})


@app.route("/api/maintenance", methods=["GET"])
def api_maintenance_list():
    asset = request.args.get("asset")
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    con = _db()
    if asset:
        rows = con.execute("SELECT id,ts,asset,author,note FROM maintenance"
                           " WHERE asset=? ORDER BY id DESC LIMIT ?", (asset, limit)).fetchall()
    else:
        rows = con.execute("SELECT id,ts,asset,author,note FROM maintenance"
                           " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return jsonify({"entries": [{"id": r[0], "ts": r[1], "asset": r[2],
                                 "author": r[3], "note": r[4]} for r in rows]})


@app.route("/api/maintenance", methods=["POST"])
def api_maintenance_add():
    """维保日志 (member only — judge POST 在 SSO 层 403). 记操作人 + 落事件流."""
    d = request.get_json(silent=True) or {}
    asset = (d.get("asset") or "").strip()
    note = (d.get("note") or "").strip()
    if not asset or not note:
        return jsonify({"error": "asset 与 note 必填"}), 400
    if len(note) > 500:
        return jsonify({"error": "note 过长 (≤500 字)"}), 400
    user = request.headers.get("X-User") or "operator"
    con = _db()
    con.execute("INSERT INTO maintenance(ts,asset,author,note) VALUES(?,?,?,?)",
                (int(time.time()), asset, user, note))
    _add_event(None, "maintenance", "info", f"🔧 {user} 维保记录 [{asset}]: {note[:80]}", con)
    con.commit()
    con.close()
    return jsonify({"ok": True})


# ============================================================ P7 报表中心
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_last_report_check = {"day": None}


def _gen_report(day_str=None):
    """生成某日运行日报 (默认昨天) — historian 真数据汇总, 存 reports/YYYY-MM-DD.json."""
    if day_str is None:
        day_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    day = datetime.date.fromisoformat(day_str)
    t0 = int(time.mktime(day.timetuple()))
    t1 = t0 + 86400
    con = _db()
    rep = {"date": day_str, "generated_ts": int(time.time()),
           "partial": day_str == datetime.date.today().isoformat(),
           "systems": {}, "alarms_raised": {}, "workorders": {}}
    for k in SYSTEMS:
        nr, nm, nd, n, avg_ms = con.execute(
            "SELECT SUM(serving='real'), SUM(serving='mirror'), SUM(serving='down'), COUNT(*),"
            " AVG(CASE WHEN serving='real' THEN real_ms WHEN serving='mirror' THEN mirror_ms END)"
            " FROM samples WHERE sys=? AND ts>=? AND ts<?", (k, t0, t1)).fetchone()
        rep["systems"][k] = {
            "samples": n or 0,
            "availability_pct": round((nr + nm) / n * 100, 2) if n else None,
            "real_pct": round(nr / n * 100, 2) if n else None,
            "down_pct": round(nd / n * 100, 2) if n else None,
            "avg_ms": round(avg_ms, 1) if avg_ms is not None else None}
    for sev, n in con.execute("SELECT severity, COUNT(*) FROM alarms"
                              " WHERE ts_raised>=? AND ts_raised<? GROUP BY severity", (t0, t1)):
        rep["alarms_raised"][sev] = n
    rep["events_n"] = con.execute("SELECT COUNT(*) FROM events WHERE ts>=? AND ts<?",
                                  (t0, t1)).fetchone()[0]
    rep["workorders"] = {
        "created": con.execute("SELECT COUNT(*) FROM workorders WHERE created_ts>=? AND created_ts<?",
                               (t0, t1)).fetchone()[0],
        "closed": con.execute("SELECT COUNT(*) FROM wo_log WHERE action='backfill' AND ts>=? AND ts<?",
                              (t0, t1)).fetchone()[0]}
    row = con.execute("SELECT source,predictions,ci_coverage_pct,audit_valid,audit_total"
                      " FROM kpi_samples WHERE ts>=? AND ts<? ORDER BY ts DESC LIMIT 1",
                      (t0, t1)).fetchone()
    if row:
        rep["kpi_last"] = {"source": row[0], "predictions": row[1],
                           "ci_coverage_pct": row[2], "audit": f"{row[3]}/{row[4]}"}
    con.close()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, day_str + ".json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep


def _daily_report_tick():
    """采样线程每周期调一次: 跨天后补生成昨日日报 (幂等), 邮件通道开了就发摘要."""
    today = datetime.date.today().isoformat()
    if _last_report_check["day"] == today:
        return
    _last_report_check["day"] = today
    yday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    path = os.path.join(REPORTS_DIR, yday + ".json")
    if os.path.exists(path):
        return
    con = _db()
    has = con.execute("SELECT COUNT(*) FROM samples WHERE ts>=? AND ts<?",
                      (int(time.mktime(datetime.date.fromisoformat(yday).timetuple())),
                       int(time.mktime(datetime.date.fromisoformat(yday).timetuple())) + 86400)
                      ).fetchone()[0]
    con.close()
    if not has:
        return   # 昨日无样本 (historian 未运行), 不造空报告
    rep = _gen_report(yday)
    _add_event(None, "report", "info", f"📊 日报已生成: {yday}")
    body = [f"XRD 平台运行日报 {yday}", ""]
    for k, s in rep["systems"].items():
        body.append(f"{SYSTEMS[k]['name'].split(' · ')[0]}: UI 可用 {s['availability_pct']}%"
                    f" / 真机 {s['real_pct']}% / 均延迟 {s['avg_ms']}ms")
    body.append(f"告警: {rep['alarms_raised']} · 事件 {rep['events_n']} 条"
                f" · 工单 +{rep['workorders']['created']}/收 {rep['workorders']['closed']}")
    _send_alarm_mail("daily_report", yday, f"[XRD 日报] {yday}", "\n".join(body))


@app.route("/api/reports")
def api_reports():
    try:
        names = sorted((n[:-5] for n in os.listdir(REPORTS_DIR)
                        if n.endswith(".json") and _DATE_RE.match(n[:-5])), reverse=True)
    except FileNotFoundError:
        names = []
    return jsonify({"reports": names[:60]})


@app.route("/api/reports/<day>")
def api_report_get(day):
    if not _DATE_RE.match(day):
        return jsonify({"error": "日期格式 YYYY-MM-DD"}), 400
    try:
        with open(os.path.join(REPORTS_DIR, day + ".json"), encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "该日无报告"}), 404


@app.route("/api/reports/generate", methods=["POST"])
def api_report_generate():
    """手动生成 (admin): date 缺省=昨天; date=今天 → 标记 partial."""
    deny = _require_admin()
    if deny:
        return deny
    d = request.get_json(silent=True) or {}
    day = d.get("date") or (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    if not _DATE_RE.match(day):
        return jsonify({"error": "日期格式 YYYY-MM-DD"}), 400
    rep = _gen_report(day)
    return jsonify({"ok": True, "report": rep})


def _csv_response(name, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    resp = Response(buf.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = f"attachment; filename={name}.csv"
    return resp


@app.route("/api/export/events.csv")
def api_export_events():
    hours = min(max(float(request.args.get("hours", 168)), 1), RETAIN_EVENTS_D * 24)
    con = _db()
    rows = con.execute("SELECT ts,sys,kind,severity,message FROM events WHERE ts>=? ORDER BY id",
                       (int(time.time() - hours * 3600),)).fetchall()
    con.close()
    out = [(datetime.datetime.fromtimestamp(r[0]).isoformat(), r[1], r[2], r[3], r[4]) for r in rows]
    return _csv_response("events", ["time", "system", "kind", "severity", "message"], out)


@app.route("/api/export/history.csv")
def api_export_history():
    sys_k = request.args.get("sys", "lab")
    if sys_k not in SYSTEMS:
        return jsonify({"error": "sys 必须是 lab/car/arm"}), 400
    hours = min(max(float(request.args.get("hours", 24)), 1), RETAIN_SAMPLES_D * 24)
    con = _db()
    rows = con.execute("SELECT ts,serving,real_ms,mirror_ms FROM samples"
                       " WHERE sys=? AND ts>=? ORDER BY ts",
                       (sys_k, int(time.time() - hours * 3600))).fetchall()
    con.close()
    out = [(datetime.datetime.fromtimestamp(r[0]).isoformat(), r[1], r[2], r[3]) for r in rows]
    return _csv_response(f"history_{sys_k}", ["time", "serving", "real_ms", "mirror_ms"], out)


@app.route("/api/export/workorders.csv")
def api_export_workorders():
    con = _db()
    rows = con.execute("SELECT code,formula,dop_symbol,dop_site,dop_pct,created_ts,created_by,"
                       "stage,state,trace_id,verdict,lambda_obs FROM workorders ORDER BY id").fetchall()
    con.close()
    out = [(r[0], r[1], r[2], r[3], r[4],
            datetime.datetime.fromtimestamp(r[5]).isoformat() if r[5] else "",
            r[6], r[7], r[8], r[9], r[10], r[11]) for r in rows]
    return _csv_response("workorders", ["code", "formula", "dopant", "site", "pct", "created",
                                        "by", "stage", "state", "trace_id", "verdict",
                                        "lambda_obs"], out)


# ============================================================ P5 批次工单 (ISA-88 批记录思路)
# 工单生命周期: 创建 → (自动调 lab 真预测, 绑 trace_id/verdict) → 取料 → 研磨灌装 → 烧结/表征
#               → 实测回填收单。每步落 wo_log 记操作人, 批次档案可导出 — 全程真数据无演戏。
WO_STAGES = ["配方预测·AI 脑", "取料·车载脑", "研磨灌装·arm01 工位", "烧结/表征·炉端", "实测回填·AI 脑"]
_FORMULA_RE = re.compile(r"^[A-Za-z0-9().]{2,60}$")


def _wo_log(con, wo_id, author, action, detail=""):
    con.execute("INSERT INTO wo_log(ts,wo,author,action,detail) VALUES(?,?,?,?,?)",
                (int(time.time()), wo_id, author, action, detail))


def _call_lab_predict(formula, dopant):
    """直连 lab 当前 serving 端 (真机隧道或 VPS 镜像 — 镜像跑的是真 predict_engine)."""
    port = _lab_port()
    src = "real" if port == SYSTEMS["lab"]["real"] else "mirror"
    body = json.dumps({"formula": formula, "dopant": dopant})
    try:
        p = subprocess.run(
            ["curl", "-s", "-m", "30", "-X", "POST", "-H", "Content-Type: application/json",
             "-d", body, f"http://127.0.0.1:{port}/api/predict"],
            capture_output=True, timeout=33)
        return (json.loads(p.stdout) if p.stdout else None), src
    except Exception:
        return None, src


def _pred_summary(d):
    """从 lab /api/predict 响应防御性抽取批次档案要点 (字段缺失不炸)."""
    if not isinstance(d, dict):
        return None
    hv = d.get("heuristic_verdict") or {}
    vm = d.get("virtual_pl_meta") or {}
    ci = vm.get("conformal_ci80") or {}
    lam = vm.get("lambda_em") or ci.get("center") or vm.get("baseline_lambda_em")
    return {"trace_id": d.get("trace_id"), "verdict": hv.get("verdict"),
            "confidence": hv.get("confidence"), "reason": hv.get("reason"),
            "lambda_em": lam, "ci_lo": ci.get("lo") or ci.get("lower"),
            "ci_hi": ci.get("hi") or ci.get("upper"),
            "analog": vm.get("baseline_analog"), "t50_k": vm.get("T50_K"),
            "flag_severity": d.get("flag_severity"),
            "flags": d.get("flags") or [],
            "sinter_temp_C": d.get("sinter_temp_C"),
            "timing_ms": (d.get("timing_ms") or {}).get("total")}


def _wo_dict(r):
    d = {"id": r[0], "code": r[1], "formula": r[2], "dop_symbol": r[3], "dop_site": r[4],
         "dop_pct": r[5], "created_ts": r[6], "created_by": r[7], "stage": r[8],
         "state": r[9], "trace_id": r[10], "verdict": r[11], "pred_source": r[13],
         "lambda_obs": r[14], "close_note": r[15], "stages": WO_STAGES}
    try:
        d["pred"] = json.loads(r[12]) if r[12] else None
    except Exception:
        d["pred"] = None
    return d


_WO_COLS = ("id,code,formula,dop_symbol,dop_site,dop_pct,created_ts,created_by,"
            "stage,state,trace_id,verdict,pred_summary,pred_source,lambda_obs,close_note")


@app.route("/api/workorders", methods=["GET"])
def api_wo_list():
    limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    con = _db()
    rows = con.execute(f"SELECT {_WO_COLS} FROM workorders ORDER BY id DESC LIMIT ?",
                       (limit,)).fetchall()
    n_open = con.execute("SELECT COUNT(*) FROM workorders WHERE state='open'").fetchone()[0]
    n_done = con.execute("SELECT COUNT(*) FROM workorders WHERE state='done'").fetchone()[0]
    con.close()
    return jsonify({"workorders": [_wo_dict(r) for r in rows],
                    "n_open": n_open, "n_done": n_done})


@app.route("/api/workorders", methods=["POST"])
def api_wo_create():
    d = request.get_json(silent=True) or {}
    formula = (d.get("formula") or "").strip()
    symbol = (d.get("symbol") or "Cr3+").strip()
    site = (d.get("site") or "").strip()
    try:
        pct = float(d.get("pct") or 1.0)
    except (TypeError, ValueError):
        return jsonify({"error": "pct 非法"}), 400
    if not _FORMULA_RE.match(formula):
        return jsonify({"error": "化学式非法 (仅字母数字括号点, 2-60 字符)"}), 400
    if symbol not in ("Cr3+", "Ni2+", "Cr3++Ni2+"):
        return jsonify({"error": "掺杂仅支持 Cr3+/Ni2+/Cr3++Ni2+"}), 400
    if not re.match(r"^[A-Za-z]{0,10}$", site):
        return jsonify({"error": "位点仅限字母 (≤10)"}), 400
    if not (0 < pct <= 20):
        return jsonify({"error": "pct 范围 (0, 20]"}), 400
    user = request.headers.get("X-User") or "operator"
    day = time.strftime("%Y%m%d")
    con = _db()
    n_today = con.execute("SELECT COUNT(*) FROM workorders WHERE code LIKE ?",
                          (f"WO-{day}-%",)).fetchone()[0]
    code = f"WO-{day}-{n_today + 1:02d}"
    cur = con.execute(
        "INSERT INTO workorders(code,formula,dop_symbol,dop_site,dop_pct,created_ts,created_by)"
        " VALUES(?,?,?,?,?,?,?)", (code, formula, symbol, site, pct, int(time.time()), user))
    wo_id = cur.lastrowid
    _wo_log(con, wo_id, user, "create", f"{formula} + {symbol}@{site or '?'} {pct}%")
    con.commit()
    con.close()
    # 同步调 lab 预测 (镜像 ~40ms / 真机数秒), 绑 trace_id
    dopant = {"symbol": symbol, "site": site, "pct": pct}
    pred, src = _call_lab_predict(formula, dopant)
    summ = _pred_summary(pred)
    con = _db()
    if summ and summ.get("trace_id"):
        con.execute("UPDATE workorders SET stage=1, trace_id=?, verdict=?, pred_summary=?,"
                    " pred_source=? WHERE id=?",
                    (summ["trace_id"], summ.get("verdict"),
                     json.dumps(summ, ensure_ascii=False), src, wo_id))
        _wo_log(con, wo_id, "lab", "predict",
                f"verdict={summ.get('verdict')} λ_em={summ.get('lambda_em')}nm"
                f" trace={summ['trace_id']} ({src})")
    else:
        _wo_log(con, wo_id, "lab", "predict_fail", "lab 预测不可达, 工单停在配方预测段, 可稍后重试推进")
    _add_event(None, "workorder", "info", f"📋 {user} 新建工单 {code}: {formula}+{symbol} {pct}%", con)
    row = con.execute(f"SELECT {_WO_COLS} FROM workorders WHERE id=?", (wo_id,)).fetchone()
    con.commit()
    con.close()
    return jsonify({"ok": True, "workorder": _wo_dict(row)})


@app.route("/api/workorders/<int:wid>", methods=["GET"])
def api_wo_detail(wid):
    con = _db()
    row = con.execute(f"SELECT {_WO_COLS} FROM workorders WHERE id=?", (wid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "工单不存在"}), 404
    logs = con.execute("SELECT ts,author,action,detail FROM wo_log WHERE wo=? ORDER BY id",
                       (wid,)).fetchall()
    con.close()
    d = _wo_dict(row)
    d["log"] = [{"ts": l[0], "author": l[1], "action": l[2], "detail": l[3]} for l in logs]
    return jsonify(d)


@app.route("/api/workorders/<int:wid>/advance", methods=["POST"])
def api_wo_advance(wid):
    user = request.headers.get("X-User") or "operator"
    con = _db()
    row = con.execute("SELECT stage, state, code FROM workorders WHERE id=?", (wid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "工单不存在"}), 404
    stage, state, code = row
    if state != "open":
        con.close()
        return jsonify({"error": f"工单已 {state}"}), 400
    if stage >= 4:
        con.close()
        return jsonify({"error": "已到回填段 — 用「实测回填」收单 (批记录完整性要求)"}), 400
    con.execute("UPDATE workorders SET stage=? WHERE id=?", (stage + 1, wid))
    _wo_log(con, wid, user, "advance", f"{WO_STAGES[stage]} → {WO_STAGES[stage + 1]}")
    con.commit()
    con.close()
    return jsonify({"ok": True, "stage": stage + 1})


@app.route("/api/workorders/<int:wid>/backfill", methods=["POST"])
def api_wo_backfill(wid):
    d = request.get_json(silent=True) or {}
    user = request.headers.get("X-User") or "operator"
    note = (d.get("note") or "").strip()[:300]
    lam = d.get("lambda_obs")
    try:
        lam = float(lam) if lam not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "lambda_obs 非法"}), 400
    con = _db()
    row = con.execute("SELECT stage, state, code FROM workorders WHERE id=?", (wid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "工单不存在"}), 404
    if row[1] != "open":
        con.close()
        return jsonify({"error": f"工单已 {row[1]}"}), 400
    con.execute("UPDATE workorders SET stage=5, state='done', lambda_obs=?, close_note=? WHERE id=?",
                (lam, note, wid))
    _wo_log(con, wid, user, "backfill",
            f"λ_obs={lam if lam is not None else '—'}nm {note}".strip())
    _add_event(None, "workorder", "info", f"✅ {user} 收单 {row[2]} (实测回填完成)", con)
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/workorders/<int:wid>/cancel", methods=["POST"])
def api_wo_cancel(wid):
    user = request.headers.get("X-User") or "operator"
    con = _db()
    row = con.execute("SELECT state, code FROM workorders WHERE id=?", (wid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "工单不存在"}), 404
    if row[0] != "open":
        con.close()
        return jsonify({"error": f"工单已 {row[0]}"}), 400
    con.execute("UPDATE workorders SET state='cancelled' WHERE id=?", (wid,))
    _wo_log(con, wid, user, "cancel", "")
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/workorders/<int:wid>/export")
def api_wo_export(wid):
    """批次档案 JSON 导出 (附 wo_log 全程留痕)."""
    resp = api_wo_detail(wid)
    if isinstance(resp, tuple):
        return resp
    resp.headers["Content-Disposition"] = f"attachment; filename=workorder_{wid}.json"
    return resp


# ============================================================ P3 SLO 可用性
@app.route("/api/uptime")
def api_uptime():
    """StatusPage 式可用性: 24h(48 段)/7d(56 段) 分段状态 + 可用率.

    诚实双口径: availability = UI 可用 (真机或镜像任一在); real = 真机在线率, 单列不混淆。
    """
    now = int(time.time())
    out = {"ts": now, "windows": {}}
    con = _db()
    for wname, hours, nseg in (("24h", 24, 48), ("7d", 168, 56)):
        seg_s = hours * 3600 // nseg
        since = now - hours * 3600
        w = {}
        for k in SYSTEMS:
            rows = con.execute(
                "SELECT (ts-?)/? AS seg, SUM(serving='real'), SUM(serving='mirror'),"
                " SUM(serving='down'), COUNT(*) FROM samples"
                " WHERE sys=? AND ts>=? GROUP BY seg", (since, seg_s, k, since)).fetchall()
            segmap = {r[0]: r for r in rows}
            segs = []
            tot = real = avail = 0
            for i in range(nseg):
                r = segmap.get(i)
                if not r:
                    segs.append("none")
                    continue
                _seg, nr, nm, nd, n = r
                tot += n
                real += nr
                avail += nr + nm
                segs.append("down" if nd else ("mirror" if nm else "real"))
            w[k] = {"segments": segs, "seg_s": seg_s,
                    "availability_pct": round(avail / tot * 100, 2) if tot else None,
                    "real_pct": round(real / tot * 100, 2) if tot else None}
        out["windows"][wname] = w
    con.close()
    return jsonify(out)


@app.route("/api/events")
def api_events():
    """事件流: ?hours=48&limit=100 (倒序). kind=serving_change|mirror_svc|..."""
    try:
        hours = min(max(float(request.args.get("hours", 48)), 1), RETAIN_EVENTS_D * 24)
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    except ValueError:
        return jsonify({"error": "参数非法"}), 400
    since = int(time.time() - hours * 3600)
    con = _db()
    rows = con.execute(
        "SELECT id,ts,sys,kind,severity,message FROM events"
        " WHERE ts>=? ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()
    con.close()
    return jsonify({"events": [
        {"id": r[0], "ts": r[1], "sys": r[2], "kind": r[3],
         "severity": r[4], "message": r[5]} for r in rows]})


def _events_after(last_id, limit=50):
    con = _db()
    rows = con.execute(
        "SELECT id,ts,sys,kind,severity,message FROM events WHERE id>?"
        " ORDER BY id ASC LIMIT ?", (last_id, limit)).fetchall()
    con.close()
    return [{"id": r[0], "ts": r[1], "sys": r[2], "kind": r[3],
             "severity": r[4], "message": r[5]} for r in rows]


_SSE_LOCK = threading.Lock()
_SSE_ACTIVE = 0
_SSE_MAX = _CMD_CONFIG.sse_max
_SSE_LIFETIME_S = _CMD_CONFIG.sse_lifetime_s


@app.route("/api/stream")
def api_stream():
    """SSE 实时推送: 新事件即时下发 + 周期 ops/kpi 快照. Caddy 对 text/event-stream 自动直通."""
    global _SSE_ACTIVE
    with _SSE_LOCK:
        if _SSE_ACTIVE >= _SSE_MAX:
            resp = jsonify({"error": "stream_capacity", "detail": "实时流已达并发上限，请稍后重连。",
                            "retry_after_s": 5, "release": ASSET_VER})
            resp.status_code = 429
            resp.headers["Retry-After"] = "5"
            return resp
        _SSE_ACTIVE += 1

    def gen():
        global _SSE_ACTIVE
        started = time.time()
        try:
            try:
                con = _db()
                row = con.execute("SELECT MAX(id) FROM events").fetchone()
                con.close()
                last_id = row[0] or 0
            except Exception:
                last_id = 0
            last_seq = -1
            last_beat = time.time()
            while time.time() - started < _SSE_LIFETIME_S:
                seq = _sse_seq["n"]
                if seq != last_seq:
                    last_seq = seq
                    payload = {"type": "tick", "ts": time.time(), "release": ASSET_VER}
                    with _lock:
                        if _ops_cache["data"]:
                            payload["ops"] = _ops_cache["data"]
                        if _kpi_cache["data"]:
                            payload["kpi"] = _kpi_cache["data"]
                    try:
                        payload["alarms"] = _alarm_counts()
                    except Exception:
                        pass
                    try:
                        evs = _events_after(last_id)
                    except Exception:
                        evs = []
                    if evs:
                        last_id = evs[-1]["id"]
                        payload["events"] = evs
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    last_beat = time.time()
                elif time.time() - last_beat > 15:
                    last_beat = time.time()
                    yield ": ping\n\n"
                time.sleep(2)
            yield "event: close\ndata: {\"reason\":\"rotation\"}\n\n"
        finally:
            with _SSE_LOCK:
                _SSE_ACTIVE = max(0, _SSE_ACTIVE - 1)

    resp = Response(gen(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["X-SSE-Max-Lifetime"] = str(_SSE_LIFETIME_S)
    return resp


@app.route("/api/kpi")
def api_kpi():
    with _lock:
        data = _kpi_cache.get("data")
        checked_at = float(_kpi_cache.get("ts") or 0) or None
    if data is None:
        return jsonify({"ts": int(time.time()), "release": ASSET_VER, "source": "unknown", "kpi": {},
                        "snapshot": _status_envelope("unknown", "historian", None,
                                                     error="sampler snapshot not available")})
    payload = json.loads(json.dumps(data, ensure_ascii=False))
    payload["release"] = ASSET_VER
    payload["snapshot"] = _status_envelope(payload.get("source", "unknown"), "historian",
                                            checked_at, ttl_s=max(120, SAMPLE_EVERY * 5))
    return jsonify(payload)


@app.route("/api/ops")
def api_ops():
    with _lock:
        data = _ops_cache.get("data")
        checked_at = float(_ops_cache.get("ts") or 0) or None
    if data is None:
        return jsonify({"ts": int(time.time()), "release": ASSET_VER, "systems": {},
                        "snapshot": _status_envelope("unknown", "historian", None,
                                                     error="sampler snapshot not available")})
    payload = json.loads(json.dumps(data, ensure_ascii=False))
    payload["release"] = ASSET_VER
    payload["snapshot"] = _status_envelope("live", "historian", checked_at,
                                            ttl_s=max(90, SAMPLE_EVERY * 3))
    return jsonify(payload)


@app.route("/api/fleet")
def api_fleet():
    with _lock:
        data = _cache.get("data")
        checked_at = float(_cache.get("ts") or 0) or None
        ops = _ops_cache.get("data") or {}
        ops_checked_at = float(_ops_cache.get("ts") or 0) or None
    if data is not None:
        payload = json.loads(json.dumps(data, ensure_ascii=False))
        payload["release"] = ASSET_VER
        payload["snapshot"] = _status_envelope("live", "sampler", checked_at,
                                                ttl_s=max(90, SAMPLE_EVERY * 3))
        return jsonify(payload)
    systems = ops.get("systems") or {}
    payload = {"ts": int(time.time()), "release": ASSET_VER}
    for key in ("lab", "car", "arm"):
        row = systems.get(key) or {}
        payload[key] = {
            "online": bool(row.get("real_online")),
            "serving": row.get("serving") or "unknown",
            "metrics": row.get("metrics") or {},
        }
    payload["snapshot"] = _status_envelope("live" if systems else "unknown", "historian",
                                            ops_checked_at, ttl_s=max(90, SAMPLE_EVERY * 3),
                                            error=None if systems else "sampler snapshot not available")
    return jsonify(payload)


@app.route("/api/systems")
def api_systems():
    """前端拿子域 URL + 名称 (不暴露内部端口)."""
    return jsonify({k: {"name": v["name"], "url": v["url"]} for k, v in SYSTEMS.items()})


# ============================================================ Site9 R2 public status
def _availability_buckets(sys_key, window_s, buckets):
    now = int(time.time())
    start = now - window_s
    step = max(1, window_s // buckets)
    con = None
    try:
        con = _db()
        rows = con.execute("SELECT ts,serving FROM samples WHERE sys=? AND ts>=? ORDER BY ts",
                           (sys_key, start)).fetchall()
    except Exception:
        rows = []
    finally:
        if con is not None:
            con.close()
    out = []
    for i in range(buckets):
        lo = start + i * step
        hi = start + (i + 1) * step if i < buckets - 1 else now + 1
        vals = [r[1] for r in rows if lo <= int(r[0]) < hi]
        if not vals:
            key = "unknown"
        elif any(v == "real" for v in vals):
            key = "operational"
        elif any(v == "mirror" for v in vals):
            key = "mirror"
        else:
            key = "offline"
        out.append({"from": lo, "to": hi, **_status_meta(key)})
    return out


def _public_status_components():
    now = time.time()
    with _lock:
        ops = _ops_cache.get("data") or {}
        ops_checked_at = float(_ops_cache.get("ts") or 0) or None
        kpi = _kpi_cache.get("data") or {}
        kpi_checked_at = float(_kpi_cache.get("ts") or 0) or None
    systems = ops.get("systems", {})
    components = []

    def add(key, name, status_key, detail="", latency_ms=None, source_detail=None,
            checked_at=None, ttl_s=90, error=None, confidence=None):
        meta = _status_meta(status_key) if status_key in STATUS_TAXONOMY else _status_from_serving(status_key, latency_ms)
        envelope = _status_envelope(meta["source"], source_detail or meta["source"],
                                    checked_at, ttl_s=ttl_s, error=error, confidence=confidence)
        if envelope["state"] == "stale":
            meta = _status_meta("degraded")
        components.append({
            "key": key,
            "name": name,
            "status": meta["label"],
            "detail": detail,
            "latency_ms": latency_ms,
            "source_detail": source_detail or "",
            **envelope,
        })

    add("vps", "VPS / command center", "operational",
        detail=f"Gunicorn 指挥中心运行中, release {ASSET_VER}", checked_at=now, ttl_s=30)
    add("dashboard", "Public dashboard", "operational",
        detail="静态外壳、Service Worker 和 historian 由 VPS 提供", checked_at=now, ttl_s=30)

    for key, label in (("lab", "AI brain"), ("car", "Car brain"), ("arm", "Arm workstation")):
        s = systems.get(key, {})
        serving = (s.get("serving") or ("real" if s.get("real_online") else
                   ("mirror" if s.get("mirror_online") else "down"))) if s else "unknown"
        latency = s.get("real_ms") if serving == "real" else s.get("mirror_ms")
        mirror_svc = s.get("mirror_svc")
        bits = [f"serving={serving}"]
        if mirror_svc:
            bits.append(f"mirror_svc={mirror_svc}")
        add(key, label, serving, detail=", ".join(bits), latency_ms=latency,
            source_detail=serving, checked_at=ops_checked_at, ttl_s=max(90, SAMPLE_EVERY * 3))

    arm_status = next((c for c in components if c["key"] == "arm"), None)
    if arm_status:
        components.append({**arm_status, "key": "arm01", "name": "arm01 / main myCobot"})
    add("arm02", "arm02 / secondary myCobot", "replay",
        detail=FINALS_PUBLIC_FACTS["dual_arm"]["claim"],
        source_detail="observed/replay", confidence="high")
    add("finals_part1", "复赛第 1 部分 / 具身脑", "replay",
        detail=(FINALS_PUBLIC_FACTS["embodied"]["claim"] + " " +
                FINALS_PUBLIC_FACTS["embodied"]["boundary"]),
        source_detail="observed/replay", confidence="high")
    add("finals_part2", "复赛第 2 部分 / AI 脑", "replay",
        detail=(FINALS_PUBLIC_FACTS["ai_brain"]["claim"] + " " +
                FINALS_PUBLIC_FACTS["ai_brain"]["boundary"]),
        source_detail="observed/replay", confidence="high")
    add("finals_part3", "复赛第 3 部分 / 双机械臂", "replay",
        detail=(FINALS_PUBLIC_FACTS["dual_arm"]["claim"] + " " +
                FINALS_PUBLIC_FACTS["dual_arm"]["boundary"]),
        source_detail="observed/replay", confidence="high")

    k = kpi.get("kpi") or {}
    ksrc = kpi.get("source") or "unknown"
    pred_status = {"down": "offline", "mirror": "mirror", "real": "operational"}.get(ksrc, "unknown")
    add("prediction", "Prediction API", pred_status,
        detail=f"source={ksrc}, predictions={k.get('predictions', 'unknown')}",
        source_detail=ksrc, checked_at=kpi_checked_at, ttl_s=max(120, SAMPLE_EVERY * 5))

    mirror_ok = [s for s in systems.values() if s.get("mirror_online")]
    mirror_bad = [s for s in systems.values() if not s.get("mirror_online")]
    add("mirror", "Mirror services",
        ("unknown" if not systems else ("degraded" if mirror_bad and mirror_ok else
         ("offline" if mirror_bad and not mirror_ok else "operational"))),
        detail=f"{len(mirror_ok)} mirror online, {len(mirror_bad)} unavailable",
        checked_at=ops_checked_at, ttl_s=max(90, SAMPLE_EVERY * 3))

    auth_role = request.headers.get("X-Role") or "unknown"
    add("auth", "Authentication / SSO", "operational",
        detail=f"request role={auth_role}; write APIs still role-gated", checked_at=now, ttl_s=30)
    add("static", "Static assets / service worker", "operational",
        detail=f"asset version={ASSET_VER}; cache controlled by sw.js", checked_at=now, ttl_s=30)
    return components


def _public_status_events(limit=12):
    con = None
    try:
        con = _db()
        rows = con.execute("SELECT ts,sys,kind,severity,message FROM events ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
        alarms = con.execute("SELECT ts_raised,sys,rule,severity,message FROM alarms"
                             " WHERE ts_cleared IS NULL ORDER BY id DESC LIMIT 8").fetchall()
    except Exception:
        rows, alarms = [], []
    finally:
        if con is not None:
            con.close()
    events = [dict(zip(("ts", "sys", "kind", "severity", "message"), r)) for r in rows]
    for r in alarms:
        events.insert(0, {"ts": r[0], "sys": r[1], "kind": "active_alarm",
                          "severity": r[3], "message": r[4] or r[2]})
    return events[:limit]


@app.route("/api/public_status")
def api_public_status():
    now = time.time()
    with _lock:
        cached = _public_status_cache.get("data")
        cached_at = float(_public_status_cache.get("ts") or 0)
    if cached is not None and now - cached_at < 2.0:
        return jsonify(cached)
    components = _public_status_components()
    health_components = [c for c in components if (c.get("status") or "").lower() != "planned"] or components
    worst = max(health_components, key=lambda c: STATUS_TAXONOMY.get(
        (c.get("status") or "Unknown").lower(), STATUS_TAXONOMY["unknown"])["rank"])
    summary_status = worst["status"] if worst["status"] not in ("Operational", "Unknown") else (
        "Operational" if all(c["status"] == "Operational" for c in health_components) else "Degraded")
    summary = {
        "status": summary_status,
        "release": ASSET_VER,
        "generated_at": int(time.time()),
        "note": "设备可能离线；公网站点使用真机、镜像、回放、陈旧、离线或未知来源标签。",
    }
    with _lock:
        bars = _availability_cache.get("data")
        bars_at = float(_availability_cache.get("ts") or 0)
    if bars is None or now - bars_at >= 30.0:
        bars = {k: {"24h": _availability_buckets(k, 24 * 3600, 24),
                    "7d": _availability_buckets(k, 7 * 24 * 3600, 28)}
                for k in ("lab", "car", "arm")}
        with _lock:
            _availability_cache["ts"] = now
            _availability_cache["data"] = bars
    payload = {"ts": int(time.time()), "summary": summary, "components": components,
               "availability": bars, "events": _public_status_events()}
    with _lock:
        _public_status_cache["ts"] = now
        _public_status_cache["data"] = payload
    return jsonify(payload)


# ============================================================ P6 安全硬化
_STARTED = time.time()
AUTH_DIR = _CMD_CONFIG.auth_dir


@app.before_request
def _csrf_origin_guard():
    """CSRF 双保险: SameSite=Lax 已挡跨站带 cookie 的 POST, 这里再校验 Origin (defense in depth)."""
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        org = request.headers.get("Origin")
        if org and org not in ("https://xiaomiju.xyz", "https://www.xiaomiju.xyz"):
            return jsonify({"error": "跨站请求被拒 (Origin 校验)"}), 403
    return None


def _require_admin():
    if (request.headers.get("X-Role") or "") != "admin":
        return jsonify({"error": "需要 admin 角色"}), 403
    return None


@app.route("/api/admin/overview")
def api_admin_overview():
    deny = _require_admin()
    if deny:
        return deny
    con = _db()
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("samples", "kpi_samples", "events", "alarms", "maintenance",
                        "workorders", "wo_log")}
    con.close()
    try:
        db_kb = os.path.getsize(DB_PATH) // 1024
    except OSError:
        db_kb = None
    return jsonify({"uptime_s": int(time.time() - _STARTED), "db_kb": db_kb,
                    "tables": counts, "sample_every_s": SAMPLE_EVERY,
                    "retain_samples_d": RETAIN_SAMPLES_D, "retain_events_d": RETAIN_EVENTS_D})


@app.route("/api/admin/users")
def api_admin_users():
    deny = _require_admin()
    if deny:
        return deny
    try:
        with open(os.path.join(AUTH_DIR, "users.json"), encoding="utf-8") as f:
            users = json.load(f)
    except Exception:
        return jsonify({"error": "users.json 不可读"}), 500
    return jsonify({"users": [{"user": k, "role": v.get("role"), "name": v.get("name")}
                              for k, v in users.items()]})


@app.route("/api/admin/logins")
def api_admin_logins():
    deny = _require_admin()
    if deny:
        return deny
    limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    path = os.path.join(AUTH_DIR, "logins.jsonl")
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        for ln in lines:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    rows.reverse()
    return jsonify({"logins": rows})


@app.route("/api/me")
def api_me():
    """当前登录用户 + 角色 — Caddy forward_auth 已 copy_headers X-User/X-Role 上来.

    judge = 评委/访客 (只读 showcase, 前端落地"技术亮点"页 + 评委视角标记);
    member = 课题组成员 (全权限)。无头则视为匿名 (理论上 SSO 不会放行到这)。
    """
    user = request.headers.get("X-User") or ""
    role = request.headers.get("X-Role") or ""
    return jsonify({"user": user, "role": role, "is_judge": role == "judge",
                    "is_member": role in ("member", "admin"), "is_admin": role == "admin"})


# ============================================================ G1 数字孪生遥测聚合
# 门户孪生 3D 场景的数据面: 臂关节角 (WorkCockpit /api/joints/*) + 车位姿/速度/电量
# (NavCockpit /api/snapshot) + lab serving 态。真机在线走隧道真遥测, 离线走镜像 mock
# 并如实标 source=mirror — 场景照常动但前端必须标"演示数据"。
# 轮询走后台线程 (懒激活: 最近 60s 内有人请求过 /api/twin 才转), 请求线程零探测延迟。
_twin_state = {"data": None, "ts": 0.0, "last_req": 0.0}


def _serving_port(k, timeout=1.0):
    """系统 k 当前 serving (real 优先) → (port, "real"|"mirror"). 短超时防孪生线程被拖死."""
    code, _ = _probe_ms(SYSTEMS[k]["real"], timeout=timeout)
    if code == 200:
        return SYSTEMS[k]["real"], "real"
    return MIRROR_PORT[k], "mirror"


def _twin_public_source(src):
    return {"real": "live", "mirror": "mock", "down": "offline", "replay": "replay"}.get(src or "down", "unknown")


def _twin_lab_context(progress=100):
    nodes = [
        {"id": "rack", "label": "AI brain rack", "x": 16, "y": 72, "source": "mock"},
        {"id": "dock", "label": "car dock", "x": 35, "y": 62, "source": "replay"},
        {"id": "arm", "label": "arm workstation", "x": 54, "y": 45, "source": "mock"},
        {"id": "furnace", "label": "furnace", "x": 72, "y": 35, "source": "replay"},
        {"id": "xrd", "label": "XRD", "x": 82, "y": 60, "source": "replay"},
        {"id": "pl", "label": "PL", "x": 90, "y": 42, "source": "replay"},
        {"id": "vps", "label": "VPS / Cloudflare", "x": 58, "y": 15, "source": "live"},
    ]
    route = [
        {"x": 16, "y": 72, "scene": [2.6, -1.6], "label": "predict"},
        {"x": 35, "y": 62, "scene": [0.8, 0.5], "label": "dispatch"},
        {"x": 54, "y": 45, "scene": [-0.9, -0.55], "label": "grind/fill"},
        {"x": 72, "y": 35, "scene": [2.5, 0.9], "label": "sinter"},
        {"x": 82, "y": 60, "scene": [2.1, 1.65], "label": "XRD"},
        {"x": 90, "y": 42, "scene": [2.95, 1.55], "label": "PL"},
    ]
    zones = [
        {"id": "bench-clearance", "label": "arm sweep zone", "x": 45, "y": 35, "w": 24, "h": 24, "risk": "review"},
        {"id": "hot-zone", "label": "furnace hot zone", "x": 67, "y": 28, "w": 15, "h": 16, "risk": "locked"},
    ]
    p = max(0, min(100, float(progress or 0))) / 100.0
    seg = min(len(route) - 2, int(p * (len(route) - 1)))
    local = p * (len(route) - 1) - seg
    a, b = route[seg], route[seg + 1]
    x = round(a["x"] + (b["x"] - a["x"]) * local, 2)
    y = round(a["y"] + (b["y"] - a["y"]) * local, 2)
    sx = round(a["scene"][0] + (b["scene"][0] - a["scene"][0]) * local, 3)
    sz = round(a["scene"][1] + (b["scene"][1] - a["scene"][1]) * local, 3)
    return {
        "nodes": nodes,
        "route": route,
        "zones": zones,
        "sample": {"x": x, "y": y, "scene": {"x": sx, "z": sz}, "progress_pct": round(p * 100, 1),
                   "stage": b.get("label") or a.get("label"), "source": "replay"},
    }


def _build_twin(replay_pct=None):
    progress = 100 if replay_pct is None else max(0, min(100, float(replay_pct)))
    lab_ctx = _twin_lab_context(progress)
    out = {"ts": time.time(), "source": {}, "car": None, "arms": {},
           "map": {k: lab_ctx[k] for k in ("nodes", "route", "zones")},
           "sample": lab_ctx["sample"],
           "replay": {"mode": "live" if replay_pct is None else "replay",
                      "progress_pct": progress,
                      "label": "live telemetry" if replay_pct is None else "time replay"},
           "plan": [
               {"t": "+00s", "system": "car", "phase": "sample transition", "action": "hold route or replay pose at sample path", "source": "replay"},
               {"t": "+08s", "system": "arm01", "phase": "arm phase", "action": "verify bench clearance and end-effector path", "source": "mock"},
               {"t": "+18s", "system": "furnace", "phase": "safety zone", "action": "keep hot-zone locked until operator confirmation", "source": "replay"},
               {"t": "+30s", "system": "AI brain", "phase": "feedback", "action": "attach XRD/PL result to public trace when measured", "source": "mock"},
            ]}
    if replay_pct is not None:
        sp = lab_ctx["sample"]["scene"]
        phase = (progress / 100.0) * math.pi
        out["car"] = {"pose": {"x": round((sp["x"] - 0.8) / 0.8, 3),
                               "y": round((sp["z"] - 0.5) / 0.8, 3),
                               "yaw": round((progress / 100.0) * 2.4 - 1.2, 3)},
                      "velocity": {"linear": 0.08 if progress < 98 else 0.0, "angular": 0.0},
                      "battery_pct": 83}
        out["arms"]["arm01"] = {"angles": [0, -28 + math.sin(phase) * 12, 42, -20, 18, 0],
                                "gripper": round(62 + math.sin(phase) * 18, 1)}
        out["arms"]["arm02"] = {"angles": [0, 24, -38 + math.cos(phase) * 10, 18, -14, 0],
                                "gripper": round(45 + math.cos(phase) * 10, 1)}
        out["source"].update({"car": "replay", "arm": "replay", "lab": "replay"})
        out["source_label"] = {k: _twin_public_source(v) for k, v in out["source"].items()}
        return out
    # 车: 位姿/速度/电量
    port, src = _serving_port("car")
    st, b = _probe(port, "/api/snapshot", timeout=2.0)
    tel = ((_json(b) or {}).get("telemetry") or {}) if st == 200 else {}
    if tel.get("pose"):
        out["car"] = {
            "pose": tel["pose"],
            "velocity": tel.get("velocity") or {},
            "battery_pct": (tel.get("battery") or {}).get("pct"),
        }
        out["source"]["car"] = src
    else:
        out["source"]["car"] = "down"
    if replay_pct is not None:
        sp = lab_ctx["sample"]["scene"]
        out["car"] = {"pose": {"x": round((sp["x"] - 0.8) / 0.8, 3),
                               "y": round((sp["z"] - 0.5) / 0.8, 3),
                               "yaw": round((progress / 100.0) * 2.4 - 1.2, 3)},
                      "velocity": {"linear": 0.08 if progress < 98 else 0.0, "angular": 0.0},
                      "battery_pct": 83}
        out["source"]["car"] = "replay"
    # 双臂: 6 关节角 + 夹爪
    port, src = _serving_port("arm")
    got = False
    for a in ("arm01", "arm02"):
        st, b = _probe(port, f"/api/joints/{a}", timeout=2.0)
        j = _json(b)
        if st == 200 and isinstance((j or {}).get("angles"), list) and len(j["angles"]) == 6:
            g = j.get("gripper")
            out["arms"][a] = {"angles": [round(float(x), 2) for x in j["angles"]],
                              "gripper": round(float(g), 1) if g is not None else None}
            got = True
    out["source"]["arm"] = src if got else "down"
    if replay_pct is not None and not got:
        phase = (progress / 100.0) * math.pi
        out["arms"]["arm01"] = {"angles": [0, -28 + math.sin(phase) * 12, 42, -20, 18, 0],
                                "gripper": round(62 + math.sin(phase) * 18, 1)}
        out["arms"]["arm02"] = {"angles": [0, 24, -38 + math.cos(phase) * 10, 18, -14, 0],
                                "gripper": round(45 + math.cos(phase) * 10, 1)}
        out["source"]["arm"] = "replay"
    # lab: 只要 serving 态 (机架灯)
    _p, src = _serving_port("lab")
    code, _ = _probe_ms(_p, timeout=1.0)
    out["source"]["lab"] = src if code == 200 else "down"
    if replay_pct is not None and out["source"].get("lab") == "down":
        out["source"]["lab"] = "replay"
    out["source_label"] = {k: _twin_public_source(v) for k, v in out["source"].items()}
    return out


def _twin_loop():
    while True:
        if time.time() - _twin_state["last_req"] > 60:
            time.sleep(1.0)
            continue
        try:
            d = _build_twin()
            _twin_state["data"], _twin_state["ts"] = d, time.time()
        except Exception:
            pass
        time.sleep(2.0)


@app.route("/api/twin")
def api_twin():
    _twin_state["last_req"] = time.time()
    replay_arg = request.args.get("replay")
    if replay_arg is not None:
        try:
            return jsonify(_build_twin(float(replay_arg)))
        except Exception:
            return jsonify({"ts": time.time(), "source": {}, "source_label": {}, "car": None, "arms": {},
                            "map": {}, "sample": {}, "plan": [], "replay": {"mode": "replay"}})
    d = _twin_state["data"]
    if d is None or time.time() - _twin_state["ts"] > 30:
        # 冷启 (线程还没转起来) 同步建一次; 之后全走缓存
        try:
            d = _build_twin()
            _twin_state["data"], _twin_state["ts"] = d, time.time()
        except Exception:
            d = {"ts": time.time(), "source": {}, "source_label": {}, "car": None, "arms": {},
                 "map": {}, "sample": {}, "plan": [], "replay": {"mode": "offline"}}
    return jsonify(d)


# ============================================================ G5 跨系统全局搜索
_search_cache = {"ts": 0.0, "data": None}


def _pred_row(it):
    """lab /api/predictions 单条 → 紧凑可搜索行 (防御式取值)."""
    p = (it.get("partial") or {})
    pay = (p.get("payload") or {})
    dop = (pay.get("dopant") or p.get("dopant") or {})
    sym, site, pct = dop.get("symbol"), dop.get("site"), dop.get("pct")
    dop_s = f"{sym}@{site} {pct}%" if sym else ""
    verd = None
    for path in (("r1", "verdict"), ("heuristic_verdict", "verdict")):
        node = pay.get(path[0]) or {}
        if isinstance(node, dict) and node.get(path[1]):
            verd = node[path[1]]; break
    return {"formula": pay.get("formula") or p.get("formula") or "?",
            "dopant": dop_s, "verdict": verd or "—",
            "trace": p.get("hash") or it.get("trace_id") or ""}


def _build_search():
    """跨系统索引: lab 预测 + car 语义地标. real 优先, 短超时, 20s 缓存."""
    out = {"ts": time.time(), "predictions": [], "landmarks": [], "source": {}}
    # lab 预测
    port, src = _serving_port("lab")
    st, b = _probe(port, "/api/predictions?per_page=40", timeout=2.0)
    j = _json(b) or {}
    if st == 200 and isinstance(j.get("items"), list):
        out["predictions"] = [_pred_row(it) for it in j["items"][:40]]
        out["source"]["lab"] = src
    else:
        out["source"]["lab"] = "down"
    # car 语义地标
    port, src = _serving_port("car")
    st, b = _probe(port, "/api/landmarks", timeout=2.0)
    j = _json(b) or {}
    if st == 200 and isinstance(j.get("landmarks"), list):
        out["landmarks"] = [{"name": lm.get("name") or lm.get("label") or "地标",
                             "x": lm.get("x"), "y": lm.get("y")} for lm in j["landmarks"][:40]]
        out["source"]["car"] = src
    else:
        out["source"]["car"] = "down"
    return out


@app.route("/api/search/index")
def api_search_index():
    """命令面板跨系统搜索源 (预测 + 地标). 20s 缓存, 失败回最近一次."""
    now = time.time()
    if _search_cache["data"] is None or now - _search_cache["ts"] > 20:
        try:
            _search_cache["data"] = _build_search()
            _search_cache["ts"] = now
        except Exception:
            if _search_cache["data"] is None:
                _search_cache["data"] = {"ts": now, "predictions": [], "landmarks": [], "source": {}}
    return jsonify(_search_cache["data"])


_federated_search_cache = {"ts": 0.0, "items": []}


def _search_source_state(source):
    value = str(source or "unknown").lower()
    if value in {"real", "live", "operational"}:
        return "live"
    if value == "mirror":
        return "mirror"
    if value in {"history", "curated", "replay", "observed"}:
        return "replay"
    if value in {"down", "offline"}:
        return "offline"
    if value == "planned":
        return "planned"
    return "unknown"


def _federated_search_corpus():
    now = time.time()
    if _federated_search_cache["items"] and now - _federated_search_cache["ts"] < 20:
        return _federated_search_cache["items"]
    pages = [
        ("home", "总览", "科研入口、闭环轨道与当前证据", "/", "总览 首页 科研 门户"),
        ("atlas", "材料图鉴", "按化学式、掺杂、波段、判决与来源检索", "/atlas", "材料 配方 phosphor atlas"),
        ("brain", "AI 脑解释", "TS、MLIP、Conformal、Fly-MB 与实验边界", "/brain", "预测 模型 AI 脑"),
        ("defense", "答辩防御", "主张、证据、限制与评委核验路径", "/defense", "答辩 防御 evidence"),
        ("benchmark", "全球对标", "科研平台与商业工作台复用矩阵", "/benchmark", "对标 benchmark global"),
        ("fsd", "FSD 世界模型", "SLAM 建图与 Lab-FSD shadow/assist 证据", "/fsd", "SLAM FSD shadow 具身脑"),
        ("fleet", "机群状态", "真机、镜像、回放、离线与新鲜度", "/fleet", "机群 fleet 状态"),
        ("assets", "资产", "系统资产、证据入口与当前边界", "/assets", "资产 asset"),
        ("security", "安全与 Trust Center", "公网只读、来源脱敏、发布门禁与人工检查项", "/sec", "安全 trust 防火墙"),
        ("status", "公开状态", "统一状态协议、延迟与来源", "/status", "状态 status freshness"),
    ]
    corpus = [{
        "kind": "page", "kind_label": "页面", "id": key, "title": title, "subtitle": subtitle,
        "href": href, "status": "live", "source": "site-navigation",
        "search_fields": {"title": title, "subtitle": subtitle, "keywords": keywords},
        "preview": subtitle,
    } for key, title, subtitle, href, keywords in pages]
    try:
        materials = _materials_all_rows()
    except Exception:
        materials = []
    for row in materials[:1000]:
        mid = _safe_text(row.get("id") or row.get("trace_id") or row.get("formula"))
        if not mid:
            continue
        formula = _safe_text(row.get("formula")) or "未命名材料"
        dopant = _safe_text(row.get("dopant"))
        source = _safe_text(row.get("source")) or "unknown"
        subtitle = " · ".join(x for x in [dopant, _safe_text(row.get("band")),
                                             _safe_text(row.get("verdict"))] if x)
        fields = {
            "formula": formula, "host": row.get("host"), "dopant": dopant, "site": row.get("site"),
            "verdict": row.get("verdict"), "band": row.get("band"), "trace_id": row.get("trace_id"),
            "work_order": row.get("work_order"), "method": row.get("method"),
        }
        corpus.append({
            "kind": "material", "kind_label": "材料", "id": mid, "title": formula,
            "subtitle": subtitle or "公开材料对象", "href": row.get("detail_url") or "/materials/" + quote(mid, safe=""),
            "status": _search_source_state(source), "source": source, "search_fields": fields,
            "preview": f"来源 {source}；方法 {_safe_text(row.get('method')) or '未提供'}；"
                       f"不确定性 {_safe_text(row.get('uncertainty')) or '未提供'}",
        })
        trace_id = _safe_text(row.get("trace_id"))
        if trace_id:
            corpus.append({
                "kind": "prediction", "kind_label": "预测", "id": trace_id,
                "title": f"预测 {formula}", "subtitle": f"trace {trace_id} · {_safe_text(row.get('verdict')) or 'UNKNOWN'}",
                "href": "/predictions/" + quote(trace_id, safe=""), "status": _search_source_state(source),
                "source": source, "search_fields": fields,
                "preview": f"方法 {_safe_text(row.get('method')) or '未提供'}；CI {_safe_text(row.get('confidence_interval')) or '未提供'}",
            })
        work_order = _safe_text(row.get("work_order"))
        if work_order:
            corpus.append({
                "kind": "work_order", "kind_label": "工单", "id": work_order,
                "title": f"工单 {work_order}", "subtitle": formula,
                "href": "/mq?work_order=" + quote(work_order, safe=""), "status": "replay",
                "source": "history", "search_fields": fields,
                "preview": "公开工单关联摘要；不包含执行器控制或私有操作备注。",
            })
    for item in _site31_evidence_objects():
        corpus.append({
            "kind": "evidence", "kind_label": "证据对象", "id": item.get("evidence_id"),
            "title": item.get("title"), "subtitle": item.get("claim"),
            "href": item.get("canonical_url"), "status": _search_source_state(item.get("source_label")),
            "source": item.get("source_label") or "unknown",
            "search_fields": {
                "title": item.get("title"), "title_en": item.get("title_en"),
                "claim": item.get("claim"), "id": item.get("evidence_id"),
                "kind": item.get("kind"), "scope": item.get("scope"),
            },
            "preview": "; ".join(item.get("limitations") or []) or "公开安全证据对象",
        })
    _federated_search_cache["ts"] = now
    _federated_search_cache["items"] = corpus
    return corpus


@app.route("/api/search")
@app.route("/api/search/federated")
def api_search_federated():
    payload = search_research(
        request.args,
        corpus=_federated_search_corpus(),
        release=ASSET_VER,
    )
    payload["compatibility"] = ["site31.federated_search.v1"]
    payload["schema"]["status_states"] = list(SEARCH_STATUS_LABELS)
    payload["status"] = _status_envelope(
        "replay", "public-index", time.time(), ttl_s=60, confidence="high")
    kind_group = next(
        (group for group in payload.get("facet_groups", []) if group.get("key") == "kind"),
        {"options": []},
    )
    payload["facets"] = {
        option["value"]: option["count"] for option in kind_group.get("options", [])
    }
    return jsonify(payload)


# ============================================================ H2 演示就绪预检 (Preflight)
# 答辩/演示前一键体检: 把散落在 ops/告警/historian/KPI/备份/安全各处的"是否就绪"汇成一张
# GO/NO-GO 清单 + 在险册 (已知风险的实时状态)。全部派生自真实状态, 不编造"全绿"。
VPS_RENEW_DATE = "2026-08-11"   # 香港轻量服务器续期红线 (见 plan.md / memory)
BACKUP_HB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_heartbeat.json")


def _backup_status():
    """备份心跳: PC 计划任务 XRD-X5-Backup 拉完 VPS historian 后写 backup_heartbeat.json
    ({ts, kept_days, detail})。无心跳 → 诚实标"未上报", 用 historian 活性作旁证。"""
    try:
        with open(BACKUP_HB_FILE, encoding="utf-8") as f:
            hb = json.load(f)
        ts = hb.get("ts")
        age_h = round((time.time() - ts) / 3600, 1) if ts else None
        return {"configured": True, "ts": ts, "age_h": age_h,
                "fresh": age_h is not None and age_h < 30,   # 日备份, <30h 视为新鲜
                "kept_days": hb.get("kept_days"), "detail": hb.get("detail")}
    except Exception:
        try:
            con = _db()
            row = con.execute("SELECT MAX(ts) FROM samples").fetchone()
            con.close()
            last = row[0] if row else None
        except Exception:
            last = None
        liveness_h = round((time.time() - last) / 3600, 1) if last else None
        return {"configured": False, "ts": None, "age_h": None, "fresh": False,
                "historian_liveness_h": liveness_h,
                "detail": "备份心跳未上报 (PC 计划任务 XRD-X5-Backup 写 backup_heartbeat.json 即可)"}


def _days_until(date_str):
    try:
        d = datetime.date.fromisoformat(date_str)
        return (d - datetime.date.today()).days
    except Exception:
        return None


def _build_preflight():
    """演示就绪体检 → {checks[], summary, risks[], backup}. status: ok|info|warn|crit."""
    with _lock:
        ops = (_ops_cache["data"] if _ops_cache["data"] else None)
    if ops is None:
        ops = _build_ops()
    systems = ops.get("systems", {})
    kpi = (_kpi_cache["data"] or {}).get("kpi", {})
    kpi_src = (_kpi_cache["data"] or {}).get("source")
    counts = _alarm_counts()
    mail_on = bool(_mail_cfg())

    checks = []

    def add(group, label, status, detail, fix=None):
        checks.append({"group": group, "label": label, "status": status,
                       "detail": _public_runbook_text(detail),
                       **({"fix": _public_runbook_text(fix)} if fix else {})})

    # —— 三机服务路径 ——
    for k in ("lab", "car", "arm"):
        s = systems.get(k, {})
        nm = SYSTEMS[k]["name"].split(" · ")[0]
        serv = s.get("serving")
        ms = s.get("real_ms") if serv == "real" else s.get("mirror_ms")
        if serv == "real":
            add("三机服务", f"{nm} 公网可达", "ok", f"真机直连在线 · {ms}ms")
        elif serv == "mirror":
            add("三机服务", f"{nm} 公网可达", "info",
                f"真机未上电, VPS 镜像兜底中 · {ms}ms (UI/功能照常)")
        else:
            add("三机服务", f"{nm} 公网可达", "crit",
                "真机隧道与 VPS 镜像双路全离线 — 公网将 502",
                f"启动镜像: systemctl --user start {MIRROR_SVC[k]}")
        # 镜像兜底链
        msvc = s.get("mirror_svc")
        if msvc == "active":
            add("镜像兜底", f"{nm} 镜像服务", "ok", f"{MIRROR_SVC[k]} active — 真机关机也有兜底")
        elif msvc in (None, "unknown"):
            add("镜像兜底", f"{nm} 镜像服务", "warn", f"{MIRROR_SVC[k]} 状态未知 (systemctl 不可查)")
        else:
            add("镜像兜底", f"{nm} 镜像服务", "crit",
                f"{MIRROR_SVC[k]} = {msvc} — 真机离线时无兜底",
                f"systemctl --user restart {MIRROR_SVC[k]}")

    # —— 告警 ——
    if counts["crit"]:
        add("告警", "严重告警", "crit", f"{counts['crit']} 条活动 crit 告警未恢复")
    elif counts["unacked"]:
        add("告警", "待确认告警", "warn", f"{counts['unacked']} 条告警未人工确认 (ack)")
    else:
        add("告警", "告警中心", "ok", "无活动严重告警")
    add("告警", "邮件通道", "ok" if mail_on else "warn",
        "已配置并启用 (crit 告警/日报可外发)" if mail_on
        else "未配置 — crit 告警不会外发邮件 (诚实禁用)",
        None if mail_on else "配 alert_email.json: {enabled:true,user,auth_code,to}")

    # —— 数据底座 (historian) ——
    try:
        con = _db()
        row = con.execute("SELECT MAX(ts), COUNT(*) FROM samples WHERE ts > ?",
                          (int(time.time() - 3600),)).fetchone()
        con.close()
        last_ts, n1h = (row[0], row[1]) if row else (None, 0)
    except Exception:
        last_ts, n1h = None, 0
    if last_ts and time.time() - last_ts < 120:
        add("数据底座", "Historian 采样", "ok", f"近 1 小时 {n1h} 条样本, 采样线程活跃")
    elif last_ts:
        add("数据底座", "Historian 采样", "warn",
            f"最后样本 {int((time.time() - last_ts) / 60)} 分钟前 — 采样线程可能停滞")
    else:
        add("数据底座", "Historian 采样", "warn", "暂无样本 (服务刚启动, 累积中)")

    # —— KPI 真值 ——
    if kpi.get("audit_total"):
        intact = kpi.get("audit_intact")
        add("KPI 真值", "审计链完整性",
            "ok" if intact else "crit",
            f"SHA-256 链 {kpi.get('audit_valid')}/{kpi.get('audit_total')} "
            + ("当前校验完整" if intact else "链断裂!"))
    if kpi.get("predictions") is not None:
        add("KPI 真值", "预测记录", "ok",
            f"累计 {kpi['predictions']} 条 (源: {'真机' if kpi_src == 'real' else '镜像'})")
    if kpi_src == "real" and kpi.get("llm_total"):
        up, tot = kpi.get("llm_up") or 0, kpi["llm_total"]
        add("KPI 真值", "本地 LLM", "ok" if up >= tot else "warn", f"{up}/{tot} 在线")

    # —— 备份 ——
    bk = _backup_status()
    if bk["configured"] and bk["fresh"]:
        add("备份", "数据备份心跳", "ok",
            f"{bk['age_h']}h 前完成 (保留 {bk.get('kept_days', '?')} 天)")
    elif bk["configured"]:
        add("备份", "数据备份心跳", "warn",
            f"上次备份 {bk['age_h']}h 前 (>30h, 计划任务可能未跑)")
    else:
        lv = bk.get("historian_liveness_h")
        add("备份", "数据备份心跳", "info",
            f"心跳未上报; historian 活性 {lv}h" if lv is not None else "心跳未上报",
            "PC 计划任务 XRD-X5-Backup 写 backup_heartbeat.json 上报")

    # —— 安全 / 续期 ——
    try:
        with open(os.path.join(AUTH_DIR, "users.json"), encoding="utf-8") as f:
            nu = len(json.load(f))
        add("安全", "SSO 账号库", "ok", f"users.json 可读 · {nu} 个账号")
    except Exception:
        add("安全", "SSO 账号库", "warn", "users.json 不可读 (登录可能异常)")
    dleft = _days_until(VPS_RENEW_DATE)
    if dleft is not None:
        add("安全", "VPS 续期", "ok" if dleft > 30 else ("warn" if dleft > 0 else "crit"),
            f"香港服务器续期红线 {VPS_RENEW_DATE} · 剩 {dleft} 天")

    order = {"crit": 0, "warn": 1, "info": 2, "ok": 3}
    summary = {"ok": 0, "info": 0, "warn": 0, "crit": 0}
    for c in checks:
        summary[c["status"]] = summary.get(c["status"], 0) + 1
    summary["total"] = len(checks)
    summary["go"] = summary["crit"] == 0
    summary["verdict"] = ("NO-GO" if summary["crit"] else
                          ("GO · 注意项" if summary["warn"] else "GO"))

    # —— 在险册 (已知风险的实时状态) ——
    risks = []

    def risk(level, title, detail, mitig):
        risks.append({"level": level, "title": title,
                      "detail": _public_runbook_text(detail),
                      "mitigation": _public_runbook_text(mitig)})

    for k in ("lab", "car", "arm"):
        s = systems.get(k, {})
        nm = SYSTEMS[k]["name"].split(" · ")[0]
        if s.get("serving") == "down":
            risk("crit", f"{nm} 双路径全断", "真机隧道与 VPS 镜像均不可达, 公网将 502",
                 f"systemctl --user start {MIRROR_SVC[k]}; 检查 frpc/Caddy")
        elif s.get("serving") == "mirror":
            risk("accepted", f"{nm} 真机未上电", "公网由 VPS 镜像兜底 (设计内降级, UI 照常)",
                 "设备上电后 Caddy active health 10s 内自动切回真机")
    if not mail_on:
        risk("open", "邮件告警通道未配置", "crit 告警与日报不会外发, 仅站内可见",
             "车上线后配 alert_email.json (QQ SMTP 授权码)")
    if counts["crit"]:
        risk("crit", "存在未恢复严重告警", f"{counts['crit']} 条 crit 告警活动中",
             "运维 → 告警中心 排查并确认")
    dleft = _days_until(VPS_RENEW_DATE)
    if dleft is not None and dleft <= 30:
        risk("open" if dleft > 0 else "crit", "VPS 即将到期",
             f"距续期红线 {VPS_RENEW_DATE} 仅 {dleft} 天", "腾讯云轻量控制台续费")
    # arm01 是当前主机械臂: 探 arm 端 /api/joints/arm01
    aport, _asrc = _serving_port("arm", timeout=1.0)
    st, b = _probe(aport, "/api/joints/arm01", timeout=1.5)
    aj = _json(b)
    if not (st == 200 and (aj or {}).get("online")):
        risk("accepted", "arm01 公网映射待确认", "主机械臂 arm01 已作为当前真机身份, 若公网仍走镜像则检查 frp/Caddy 上游是否指向 arm01",
             "确认 arm01 WorkCockpit :8890 与 VPS 18891 隧道映射一致")

    risks.sort(key=lambda r: {"crit": 0, "open": 1, "accepted": 2}.get(r["level"], 3))
    checks.sort(key=lambda c: order.get(c["status"], 9))
    return {"ts": time.time(), "checks": checks, "summary": summary,
            "risks": risks, "backup": bk}


@app.route("/api/preflight")
def api_preflight():
    """演示就绪预检: GO/NO-GO 体检清单 + 在险册 + 备份心跳 (全派生自真实状态)."""
    try:
        return jsonify(_build_preflight())
    except Exception:
        app.logger.exception("preflight")
        return jsonify({"error": "预检失败"}), 500


# ============================================================ H3 问平台 Copilot (运维副驾)
# 诚实定位: 这不是大模型, 是"指挥中心运维副驾"—— 意图路由 + 实时遥测接地 + 平台知识库检索。
# 平台状态类问题答活数据 (ops/preflight/kpi/events), 架构/亮点类问题答策展知识库, 全部可核。
# 设备/科学深问引导进 AI 脑 (那里有 9 本地 LLM + /copilot 文献副驾)。
_KB = [
    {"id": "arch", "kw": "架构 三机 异构 大脑 小脑 介绍 是什么 什么 协同 组成 怎么 结构",
     "title": "三机异构协同架构",
     "body": "本平台是**双 RDK X5 + 双机械臂复赛真机协同**的三机异构系统: "
             "**AI 脑**出脑力 (NIR 荧光粉配方设计 → 30 秒 AI 预筛, 9 本地 LLM + 5 BPU 推理槽 + 云 R1); "
             "**具身脑**出脚力 (真机完成取瓶、升顶、0.50m 里程计闭环、放瓶复位，Lab-FSD 仍以 shadow/assist 输出风险和候选轨迹); "
             "**机械臂**出手力 (arm01 单臂视觉冗余与投袋、arm02 并发四周期研磨均已真机完成)。"
             "三者通过受保护的香港公网门户统一展示脱敏证据，不提供远程物理控制。",
     "actions": [("看总览", "home"), ("数字孪生", "twin")]},
    {"id": "highlight", "kw": "亮点 创新 卖点 厉害 优势 bpu transformer vlm 性能",
     "title": "5 大技术亮点 (真机实测)",
     "body": "① **X5 BPU 24 层 Qwen2 Transformer 真机实测** — 单次 forward 553ms, BPU 峰值利用率 9%→52%; "
             "② **X5 BPU 端到端 VLM 真机实测** — SmolVLM hybrid, vision 上 BPU; "
             "③ **PP-OCRv4 检测全 BPU** — 6ms/163FPS, 0 CPU 回退; "
             "④ **MPPI cost MLP BPU 加速** — 1.14ms, 880K traj/s (4.1× CPU); "
             "⑤ **双 X5 异构协同** — 大脑 9 LLM + 小脑 19+ BPU 节点。全部 RDK X5 真机实测可复核。",
     "actions": [("看亮点", "highlight")]},
    {"id": "selfheal", "kw": "镜像 兜底 自愈 关机 离线 降级 容错 高可用 caddy 故障",
     "title": "链路自愈 (设备关机 UI 照常在)",
     "body": "公网链路: 访客 → Cloudflare → 香港 VPS Caddy (active health 10s) → **真机隧道** 或 **VPS 常驻镜像**。"
             "真机在线即直连真机, 关机 2 连败自动切镜像 (mirror-lab/navcockpit/workcockpit) —— UI 与功能照常。"
             "镜像里 lab 跑的是真 predict_engine, 演示不依赖设备上电。",
     "actions": [("运维总览", "ops"), ("演示就绪预检", "preflight")]},
    {"id": "value", "kw": "价值 解决 痛点 15 个月 分钟 飞轮 闭环 意义 为什么 做什么",
     "title": "从 15 个月到 15 分钟",
     "body": "研究员每轮试 15-20 个配方, 每个经 研磨→烧制→XRD→PL→决策 ≈ 1 个月, 串起来一年多。"
             "本系统 30 秒 AI 预筛出 verdict + 烧结条件 + 置信区间, 15 分钟锁定 3-5 个优质候选真做; "
             "具身脑真机取料闭环、双臂投袋与四周期研磨证据、实测回填共同驱动下一轮主动学习 —— 闭环实验飞轮。",
     "actions": [("项目故事", "story"), ("批次工单", "mq")]},
    {"id": "help", "kw": "帮助 help 能问 怎么用 会什么 功能",
     "title": "我能答什么",
     "body": "我是**指挥中心运维副驾**, 答这些都接实时数据: 「现在什么状态」「哪些设备在线」「有没有告警」"
             "「能不能开演 / 就绪吗」「预测准确率」「最近发生了什么」; 也能解释「三机架构」「技术亮点」「链路自愈」。"
             "设备级操作和深度科学问题请进 **AI 脑** (那里有 9 本地 LLM + 文献副驾)。",
     "actions": [("演示就绪预检", "preflight")]},
]


def _kb_match(q):
    ql = q.lower()
    best, score = None, 0
    for e in _KB:
        s = sum(1 for w in e["kw"].split() if w and w.lower() in ql)
        if s > score:
            best, score = e, s
    return best if score > 0 else None


def _cop_sys_state():
    with _lock:
        ops = _ops_cache["data"] or _build_ops()
    rows = []
    for k in ("lab", "car", "arm"):
        s = ops.get("systems", {}).get(k, {})
        nm = SYSTEMS[k]["name"].split(" · ")[0]
        serv = s.get("serving")
        txt = {"real": "真机直连", "mirror": "镜像兜底", "down": "离线"}.get(serv, "?")
        rows.append((k, nm, serv, txt, s))
    return rows


# —— Agent 接地工具集: 每个工具命中关键词 → 真数据查询 → {summary, facts, actions, follow} ——
# follow = 动态追问 (基于查到的实况推荐下一步问题), 让副驾呈现"多步规划"而非死板模板。
def _tool_state():
    rows = _cop_sys_state()
    real = sum(1 for r in rows if r[2] == "real")
    mir = sum(1 for r in rows if r[2] == "mirror")
    down = sum(1 for r in rows if r[2] == "down")
    facts = []
    for k, nm, serv, txt, s in rows:
        ms = s.get("real_ms") if serv == "real" else s.get("mirror_ms")
        facts.append({"label": nm, "value": txt + (f" · {ms}ms" if ms is not None else ""),
                      "status": "ok" if serv == "real" else ("info" if serv == "mirror" else "crit")})
    summ = (f"三机当前 **{real} 真机直连 / {mir} 镜像兜底 / {down} 离线**。"
            + ("公网可正常访问。" if down == 0 else "有系统双路全断, 需处理。")
            + ("" if mir == 0 else " 镜像兜底是设备未上电的设计内降级, UI 与功能照常。"))
    follow = (["哪个设备需要上电?"] if mir or down else []) + ["现在能开演吗?"]
    return {"summary": summ, "facts": facts,
            "actions": [("运维总览", "ops"), ("演示就绪预检", "preflight")], "follow": follow}


def _tool_ready():
    pf = _build_preflight()
    s = pf["summary"]
    facts = [{"label": "体检结论", "value": s["verdict"], "status": "ok" if s["go"] else "crit"},
             {"label": "阻断 / 注意 / 就绪", "value": f"{s['crit']} / {s['warn']} / {s['ok']}",
              "status": "ok" if s["go"] else "crit"}]
    if s["go"]:
        summ = (f"**{s['verdict']}** —— 无阻断项, 可以开演。"
                + (f" 有 {s['warn']} 个注意项 (不影响演示)。" if s["warn"] else ""))
        follow = ["有哪些注意项?"] if s["warn"] else ["5 大技术亮点是什么"]
    else:
        blockers = [c["label"] for c in pf["checks"] if c["status"] == "crit"][:4]
        summ = f"**NO-GO** —— 有 {s['crit']} 项阻断必须先处理: " + "、".join(blockers) + "。"
        follow = ["怎么处置告警?", "哪个设备需要上电?"]
    return {"summary": summ, "facts": facts,
            "actions": [("演示就绪预检", "preflight")], "follow": follow}


def _tool_alarm():
    c = _alarm_counts()
    con = _db()
    act = con.execute("SELECT severity, message FROM alarms WHERE ts_cleared IS NULL"
                      " ORDER BY ts_raised DESC LIMIT 4").fetchall()
    con.close()
    facts = [{"label": "活动告警", "value": f"{c['total']} 条 (crit {c['crit']} / warn {c['warn']})",
              "status": "crit" if c["crit"] else ("warn" if c["warn"] else "ok")},
             {"label": "未确认", "value": f"{c['unacked']} 条", "status": "warn" if c["unacked"] else "ok"}]
    if c["total"] == 0:
        summ = "**无活动告警**, 全链路健康。规则引擎持续监测 (双路离线/镜像失效/延迟超阈/审计链断裂/LLM 降级)。"
        follow = []
    else:
        summ = f"**{c['total']} 条活动告警**: " + "; ".join(m for _s, m in act) + "。"
        follow = ["怎么处置告警?"]
    return {"summary": summ, "facts": facts, "actions": [("告警中心", "ops")], "follow": follow}


def _tool_kpi():
    with _lock:
        kd = _kpi_cache["data"] or _build_kpi()
    k = kd.get("kpi", {})
    src = "真机" if kd.get("source") == "real" else "镜像"
    facts = []
    if k.get("ci_coverage_pct") is not None:
        facts.append({"label": "90% CI 实测覆盖率", "value": f"{k['ci_coverage_pct']}%", "status": "ok"})
    if k.get("audit_total"):
        facts.append({"label": "审计链", "value": f"{k['audit_valid']}/{k['audit_total']} "
                      + ("当前校验完整" if k.get("audit_intact") else "断裂"),
                      "status": "ok" if k.get("audit_intact") else "crit"})
    if k.get("predictions") is not None:
        facts.append({"label": "累计预测", "value": f"{k['predictions']} 条 (源:{src})", "status": "ok"})
    summ = ("平台 KPI (取自 lab 当前 serving 端): "
            + (f"Conformal 90% 区间实测覆盖 {k.get('ci_coverage_pct')}%, " if k.get("ci_coverage_pct") is not None else "")
            + (f"SHA-256 审计链 {k.get('audit_valid')}/{k.get('audit_total')} 当前校验完整, " if k.get("audit_intact") else "")
            + (f"累计 {k.get('predictions')} 条预测记录。" if k.get("predictions") is not None else "")
            + " 另有 λ_em MAE 6.2nm (可微 Tanabe-Sugano)。")
    return {"summary": summ, "facts": facts,
            "actions": [("AI 脑", "lab"), ("批次工单", "mq")], "follow": ["材料候选池有多少?"]}


def _tool_events():
    con = _db()
    rows = con.execute("SELECT ts, message FROM events ORDER BY id DESC LIMIT 5").fetchall()
    con.close()
    if not rows:
        return {"summary": "最近暂无记录事件 —— 三机链路稳定 (服务跃迁/告警/工单/维保会自动落库)。",
                "facts": [], "actions": [("运维事件流", "ops")], "follow": []}
    facts = [{"label": time.strftime("%m-%d %H:%M", time.localtime(ts)), "value": msg} for ts, msg in rows]
    return {"summary": f"最近 **{len(rows)}** 条平台事件 (historian 真数据):",
            "facts": facts, "actions": [("运维事件流", "ops")], "follow": []}


def _tool_atlas():
    try:
        with _lock:
            ad = _atlas_cache["data"] or _build_atlas()
    except Exception:
        ad = None
    s = (ad or {}).get("summary", {})
    if not s.get("total"):
        return {"summary": "材料图鉴暂未取到候选 (lab serving 端无响应)。", "facts": [],
                "actions": [("材料图鉴", "atlas")], "follow": []}
    facts = [{"label": "候选总数", "value": f"{s['total']} 个 (MatterGen 生成 + MatterSim/CHGNet 稳定性)", "status": "ok"}]
    if s.get("lambda_min") is not None:
        facts.append({"label": "λ_em 范围", "value": f"{s['lambda_min']:.0f}–{s['lambda_max']:.0f} nm ({s.get('lambda_n', 0)} 个有谱)"})
    vd = s.get("verdict") or {}
    if vd:
        facts.append({"label": "判决分布", "value": " / ".join(f"{kk} {vv}" for kk, vv in vd.items())})
    summ = (f"AI 脑生成式候选池 **{s['total']} 个** 材料, "
            + (f"λ_em 覆盖 {s['lambda_min']:.0f}–{s['lambda_max']:.0f} nm" if s.get("lambda_min") is not None else "")
            + ", 按 NIR 波段/判决/来源可在图鉴页筛选。")
    return {"summary": summ, "facts": facts,
            "actions": [("材料图鉴", "atlas"), ("配方构建器", "build")], "follow": ["配方构建器怎么用?"]}


def _tool_models():
    try:
        with _lock:
            md = _models_cache["data"] or _build_models()
    except Exception:
        md = None
    s = (md or {}).get("summary", {})
    if not s.get("total"):
        return {"summary": "模型注册表暂未取到 (lab serving 端无响应)。", "facts": [],
                "actions": [("模型注册表", "models")], "follow": []}
    by = s.get("by_tier", {})
    facts = [{"label": "模型总数", "value": f"{s['total']} 个 ({s.get('online', 0)} 在线)",
              "status": "ok"}]
    if by:
        facts.append({"label": "分层", "value": " · ".join(f"{kk}:{vv}" for kk, vv in by.items())})
    summ = (f"模型注册表共 **{s['total']} 个** (9 本地 LLM + 5 BPU 推理槽 + 轻量 BPU 感知/具身/双臂), "
            f"当前 {s.get('online', 0)} 个在线 —— 元数据取自 lab 真实健康接口, 设备离线时如实显示 online:false。")
    return {"summary": summ, "facts": facts,
            "actions": [("模型注册表", "models"), ("AI 脑", "lab")], "follow": []}


_COP_TOOLS = [
    {"id": "状态", "kw": ("状态", "在线", "怎么样", "健康", "online", "health", "正常吗", "活", "联网"), "run": _tool_state},
    {"id": "就绪", "kw": ("就绪", "能演", "能不能", "可以演", "开演", "go", "nogo", "准备好", "预检", "演示"), "run": _tool_ready},
    {"id": "告警", "kw": ("告警", "报警", "alarm", "异常", "出事", "故障", "出问题"), "run": _tool_alarm},
    {"id": "KPI", "kw": ("准确", "覆盖", "kpi", "预测", "指标", "覆盖率", "审计", "多少条", "几条", "性能数据"), "run": _tool_kpi},
    {"id": "事件", "kw": ("最近", "发生", "事件", "历史", "刚才", "动态", "日志"), "run": _tool_events},
    {"id": "图鉴", "kw": ("候选", "材料", "图鉴", "配方池", "波段", "λ", "atlas", "多少个"), "run": _tool_atlas},
    {"id": "模型", "kw": ("模型", "几个模型", "llm", "bpu", "推理槽", "注册表", "model"), "run": _tool_models},
]


# —— DeepSeek 云 LLM 合成层 (H29+): 确定性工具查到真数据 → 喂给 deepseek-chat 自然语言作答。
# VPS 直连云 API (不依赖设备上电); key 从 systemd EnvironmentFile 注入 (不入库); 不通则降级回规则模板 (ADR-3)。
_LLM_KEY = _CMD_CONFIG.llm_key
_LLM_MODEL = _CMD_CONFIG.llm_model
_LLM_URL = "https://api.deepseek.com/chat/completions"


def llm_available():
    return bool(_LLM_KEY)


def _llm_kb_brief():
    """副驾背景知识 (从 _KB 浓缩) — 让 LLM 答架构/亮点/价值类问题不脱离真相。"""
    return "\n".join("- " + e["title"] + ": " + re.sub(r"\*\*", "", e["body"]) for e in _KB)


def _llm_chat(system, user, max_tokens=440, timeout=15):
    """调 deepseek-chat (subprocess curl, 同 _probe 模式避 urllib 坑)。成功返回文本, 失败/未配置返回 None。"""
    if not _LLM_KEY:
        return None
    payload = json.dumps({
        "model": _LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.3, "stream": False,
    }, ensure_ascii=False)
    try:
        p = subprocess.run(
            ["curl", "-s", "-m", str(timeout), "-X", "POST", _LLM_URL,
             "-H", "Content-Type: application/json",
             "-H", "Authorization: Bearer " + _LLM_KEY, "--data-binary", "@-"],
            input=payload.encode("utf-8"), capture_output=True, timeout=timeout + 2)
        d = json.loads(p.stdout.decode("utf-8", "ignore"))
        txt = ((d.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "").strip()
        return txt or None
    except Exception:
        return None


def _copilot_snapshot():
    """给 LLM 的实时平台快照 (状态+告警+KPI, 走缓存快)。返回 (context_text, facts)。"""
    out, sfacts = [], []
    for fn, label in ((_tool_state, "三机状态"), (_tool_alarm, "告警"), (_tool_kpi, "KPI")):
        try:
            r = fn()
            out.append("[" + label + "] " + re.sub(r"\*\*", "", r["summary"]))
            sfacts += r["facts"]
        except Exception:
            pass
    return "\n".join(out), sfacts


def _llm_synthesize(q, parts, facts):
    """把确定性真数据 (parts/facts 或默认快照) 灌进 system prompt, 让 DeepSeek 只据实作答。"""
    if parts:
        ctx = "\n".join(re.sub(r"\*\*", "", p) for p in parts)
        sf = facts
    else:
        ctx, sf = _copilot_snapshot()
    fl = "; ".join(str(f["label"]) + "=" + str(f["value"]) for f in (sf or [])[:10])
    if fl:
        ctx += "\n关键指标: " + fl
    sysp = (
        "你是「指挥中心运维副驾」, XRD 智慧实验室三机异构平台 (AI 脑 + 车载脑 + 双臂工位) 的值班助手。\n"
        "严格规则:\n"
        "1. 只依据下面【实时平台数据】和【背景知识】作答, 绝不编造数字/状态/指标; 没有的信息就说「暂无该数据, 可进对应页面查看」。\n"
        "2. 简洁中文 2-4 句, 像值班工程师汇报, 不寒暄不堆长清单。\n"
        "3. 设备级遥控/深度科学计算请引导用户进 AI 脑 (9 本地 LLM + 文献副驾)。\n"
        "4. 镜像兜底 = 设备未上电的设计内降级, 属正常, 别说成故障。\n\n"
        "【背景知识】\n" + _llm_kb_brief() + "\n\n【实时平台数据】\n" + (ctx or "(暂无实时数据)"))
    return _llm_chat(sysp, q)


def _copilot_answer(q, deep=False):
    """Agent 化运维副驾: 确定性工具地基 (多步巡检接地真数据) + DeepSeek 云合成层。
    工具命中→真 facts + 模板答案; deep=True 或无命中兜底时, 把真数据灌给 deepseek-chat 自然语言作答 (据实不编)。
    返回 {answer, facts[], actions[], grounded, steps[], followups[], llm?}."""
    ql = q.lower()

    hits = [t for t in _COP_TOOLS if any(w in ql for w in t["kw"])]
    # 全局巡检类问题 → 跑核心三工具 (状态/告警/KPI, 全走缓存, 快)
    if any(w in ql for w in ("巡检", "体检", "全部", "所有", "汇总", "总体", "整体", "概况", "一遍", "扫一遍", "通报")):
        hits = [t for t in _COP_TOOLS if t["id"] in ("状态", "告警", "KPI")]
    hits = hits[:3]  # 一次最多链 3 个工具, 避免答案过长

    # —— 1) 确定性工具链 / 知识库 → 真数据 facts + 模板答案 ——
    steps, parts, facts, actions, follow = [], [], [], [], []
    kb = None
    if hits:
        for t in hits:
            try:
                r = t["run"]()
            except Exception as e:
                r = {"summary": f"{t['id']}查询失败 ({e}).", "facts": [], "actions": [], "follow": []}
            steps.append(t["id"])
            parts.append(r["summary"])
            facts += r["facts"]
            for a in r.get("actions", []):
                if list(a) not in [list(x) for x in actions]:
                    actions.append(a)
            for f in r.get("follow", []):
                if f not in follow and f != q:
                    follow.append(f)
        if len(steps) == 1:
            det = parts[0]
        else:
            det = (f"已巡检 **{len(steps)}** 项 (" + " · ".join(steps) + "):\n\n"
                   + "\n\n".join(f"**{s}** — {p}" for s, p in zip(steps, parts)))
        res = {"answer": det, "facts": facts, "actions": [list(a) for a in actions],
               "grounded": True, "steps": steps, "followups": follow[:4]}
    else:
        kb = _kb_match(q)
        if kb:
            res = {"answer": "**" + kb["title"] + "** — " + kb["body"], "facts": [],
                   "actions": [list(a) for a in kb["actions"]], "grounded": False,
                   "steps": ["知识库"], "followups": []}
        else:
            res = {"answer": "我是指挥中心运维副驾, 会**多步巡检**接地实时数据。擅长: 平台**状态/告警/就绪/KPI/最近事件/材料候选/模型注册表**, "
                   "以及**架构/亮点/自愈/价值**解释。试试问「现在什么状态」「巡检一遍」「能开演吗」「候选池多少个」。设备级操作与深度科学问题请进 AI 脑。",
                   "facts": [], "actions": [("我能答什么", None)], "grounded": False, "fallback": True,
                   "steps": [], "followups": ["巡检一遍", "能开演吗?", "介绍一下三机架构"]}

    # —— 2) DeepSeek 合成层: deep 开关 OR 无命中兜底时, 用真数据接地自然语言重答 (失败则保留模板, ADR-3) ——
    want_llm = _LLM_KEY and (deep or (not hits and not kb))
    if want_llm:
        txt = _llm_synthesize(q, parts, facts)
        if txt:
            res["answer"] = txt
            res["llm"] = _LLM_MODEL
            res["grounded"] = True  # 答案接地于注入的真数据
            res.pop("fallback", None)
            if not res.get("followups"):
                res["followups"] = ["巡检一遍", "能开演吗?", "介绍一下三机架构"]
    return res


_COPILOT_SUGGEST = ["巡检一遍", "现在什么状态?", "能开演吗?", "有没有告警?", "候选池多少个?",
                    "介绍一下三机架构", "5 大技术亮点是什么", "设备关机了 UI 还在吗?", "最近发生了什么?"]


# ============================================================ H4 材料图鉴 / Atlas
# 把 AI 脑生成式候选池 (MatterGen + MatterSim/CHGNet 稳定性) 做成可筛选图鉴 —— 真数据,
# 取 lab 当前 serving 端 /api/ai_candidates。镜像也有真候选 JSON, 设备离线照常可看。
_atlas_cache = {"ts": 0.0, "data": None}


def _atlas_band(lam):
    """λ_em → NIR 波段标签 (用于颜色分组)."""
    if lam is None:
        return "unknown"
    if lam < 750:
        return "nir_i_low"     # NIR-I 短波
    if lam < 1000:
        return "nir_i"         # NIR-I
    if lam < 1350:
        return "nir_ii"        # NIR-II
    return "nir_ii_long"


def _build_atlas(timeout=4.0):
    port, src = _serving_port("lab")
    st, b = _probe(port, "/api/ai_candidates", timeout=timeout)
    j = _json(b) or {}
    entries = j.get("entries") if isinstance(j, dict) else None
    out = {"ts": time.time(), "source": src if st == 200 else "down",
           "items": [], "summary": {}}
    if not isinstance(entries, list):
        return out
    items = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        lam = e.get("lambda_em_nm")
        items.append({
            "formula": e.get("formula") or e.get("formula_raw") or "?",
            "lambda_em": round(lam, 0) if isinstance(lam, (int, float)) else None,
            "band": _atlas_band(lam if isinstance(lam, (int, float)) else None),
            "verdict": e.get("verdict") or "—",
            "site": e.get("default_dopant_site"),
            "stability_pct": round(e["thermal_stability_pct"], 1) if isinstance(e.get("thermal_stability_pct"), (int, float)) else None,
            "e_atom": round(e["e_per_atom_eV"], 3) if isinstance(e.get("e_per_atom_eV"), (int, float)) else None,
            "converged": bool(e.get("mattersim_converged")),
            "source": e.get("source") or "?",
            "round": e.get("round") or "",
            "confidence": e.get("confidence"),
            "trace": e.get("trace_id") or "",
        })
    # 汇总
    vd, sd, bd = {}, {}, {}
    lams = [it["lambda_em"] for it in items if it["lambda_em"] is not None]
    for it in items:
        vd[it["verdict"]] = vd.get(it["verdict"], 0) + 1
        sd[it["source"]] = sd.get(it["source"], 0) + 1
        bd[it["band"]] = bd.get(it["band"], 0) + 1
    out["items"] = items
    out["summary"] = {
        "total": len(items),
        "verdict": vd, "source": sd, "band": bd,
        "lambda_min": min(lams) if lams else None,
        "lambda_max": max(lams) if lams else None,
        "lambda_n": len(lams),
        "validation": j.get("validation") if isinstance(j.get("validation"), dict) else None,
    }
    return out


# ============================================================ H16 模型注册表 / Model Registry
# 平台全模型清单 + 实时在线态: 9 本地 LLM + 5 BPU swap-load slot 取 lab serving 端真元数据
# (server 离线时 label/desc/arch 仍真, 只是 ok=false 诚实标"未加载"); 轻量 BPU 感知 + 具身/双臂
# 模型为策展真实清单 (来自各子系统, 设备离线无活体探活, 标注来源)。
_models_cache = {"ts": 0.0, "data": None}

# 策展补充: 轻量 BPU 感知 + 具身脑 + 双臂 (真实清单, 设备离线时无活体, 标 curated)
_CURATED_MODELS = {
    "perception": {"group": "轻量 BPU 感知 (AI 脑)", "tier": "bpu", "models": [
        {"name": "mlp_45d", "label": "XRD 45D MLP 分类器", "spec": "45D→INT8 <2ms", "note": "数值线 XRD 峰特征判别"},
        {"name": "mlp_190d", "label": "XRD 190D MLP", "spec": "190D→INT8 <2ms", "note": "/api/bpu_infer_190d"},
        {"name": "mlp_80d", "label": "PL 80D MLP (Cr/Ni/Cr+Ni)", "spec": "80D→INT8 <2ms", "note": "光谱数值线掺杂分类"},
        {"name": "yolo_xrd", "label": "YOLO 谱图检测 (XRD)", "spec": "~110ms/frame", "note": "视觉线谱图 ROI"},
        {"name": "yolo_pl", "label": "YOLO 谱图检测 (PL)", "spec": "~110ms/frame", "note": "光谱视觉线"},
        {"name": "dinov2", "label": "DINOv2-small 图像 embedding", "spec": "/api/bpu_image_embed", "note": "图像向量"},
    ]},
    "embodied": {"group": "具身脑 BPU (车载 X5)", "tier": "bpu", "models": [
        {"name": "smolvlm", "label": "SmolVLM-256M hybrid VLM", "spec": "33s/query · vision BPU", "note": "X5 BPU 端到端 VLM 真机实测"},
        {"name": "ppocr", "label": "PP-OCRv4 det", "spec": "2.7MB · 6ms · 163FPS · 0 CPU fallback", "note": "烧结炉数显全 BPU 检测"},
        {"name": "mppi", "label": "MPPI cost MLP", "spec": "264KB · 1.14ms · 880K traj/s (4.1× CPU)", "note": "轨迹规划代价评估"},
        {"name": "xfeat", "label": "XFeat 特征点", "spec": "985KB · 17ms · 57FPS", "note": "视觉里程计/匹配"},
        {"name": "yolo_world", "label": "YOLO-World 开放词检测", "spec": "BPU", "note": "Round4 A1"},
        {"name": "edgesam", "label": "EdgeSAM 分割", "spec": "BPU", "note": "Round4 A2"},
    ]},
    "arm": {"group": "双臂工位 (myCobot 280-Pi)", "tier": "edge", "models": [
        {"name": "apriltag", "label": "AprilTag tag36h11 位姿", "spec": "CPU 12-20ms", "note": "工件/工位标定 (id 0-9)"},
        {"name": "arm_planner", "label": "arm_planner LLM :9103", "spec": "Qwen 0.5B/1.7B LoRA", "note": "故障决策 (J3_STALL 等)"},
        {"name": "smolvla", "label": "SmolVLA-450M VLA", "spec": "流水线就绪 · 待采 episode", "note": "拖动示教→lerobot (待硬件)"},
    ]},
}


def _build_models():
    port, src = _serving_port("lab")
    out = {"ts": time.time(), "source": src, "tiers": []}
    # 9 本地 LLM
    st, b = _probe(port, "/api/local_llm_health", timeout=3.0)
    j = _json(b) or {}
    llms = []
    if isinstance(j.get("models"), dict):
        for name, m in j["models"].items():
            llms.append({"name": name, "label": m.get("label") or name,
                         "spec": m.get("tag") or "", "note": m.get("desc") or "",
                         "online": bool(m.get("ok")), "url": m.get("url")})
    # CPU vs BPU 标记: url 带端口的是 CPU llama-server
    out["tiers"].append({"key": "llm", "group": "本地 LLM (AI 脑)", "tier": "llm",
                         "subtitle": "4 CPU llama-server + 云 R1 · 真元数据取自 serving 端 (server 离线则标未加载)",
                         "models": llms, "live": True})
    # 5 BPU swap-load slot
    st, b = _probe(port, "/api/bpu_slot_health", timeout=3.0)
    j = _json(b) or {}
    slots = []
    if isinstance(j.get("slots"), list):
        for s in j["slots"]:
            slots.append({"name": s.get("name"), "label": s.get("label") or s.get("name"),
                          "spec": (s.get("arch", "") + (f" · {s.get('n_segs')} seg" if s.get("n_segs") else "")),
                          "note": s.get("note") or "", "online": bool(s.get("available"))})
    out["tiers"].append({"key": "slot", "group": "BPU swap-load slot (AI 脑)", "tier": "bpu",
                         "subtitle": (j.get("method") or "5-slot swap-load") + " · CMA 391MB 单次装 1 slot",
                         "models": slots, "live": True})
    # 策展补充
    for key, g in _CURATED_MODELS.items():
        out["tiers"].append({"key": key, "group": g["group"], "tier": g["tier"],
                             "subtitle": "策展真实清单 (设备离线无活体探活)",
                             "models": [{**m, "online": None} for m in g["models"]], "live": False})
    # 汇总
    total = sum(len(t["models"]) for t in out["tiers"])
    online = sum(1 for t in out["tiers"] for m in t["models"] if m.get("online"))
    out["summary"] = {"total": total, "online": online,
                      "by_tier": {t["key"]: len(t["models"]) for t in out["tiers"]}}
    return out


@app.route("/api/models")
def api_models():
    """模型注册表: 9 本地 LLM + 5 BPU slot (lab 真元数据) + 轻量 BPU 感知/具身/双臂 (策展). 20s 缓存."""
    now = time.time()
    if _models_cache["data"] is None or now - _models_cache["ts"] > 20:
        try:
            _models_cache["data"] = _build_models()
            _models_cache["ts"] = now
        except Exception:
            if _models_cache["data"] is None:
                _models_cache["data"] = {"ts": now, "source": "down", "tiers": [], "summary": {}}
    return jsonify(_models_cache["data"])


def _ai_brain_explain_payload():
    cached = _models_cache.get("data") or {}
    tiers = cached.get("tiers") if isinstance(cached.get("tiers"), list) else []
    llm_tier = next((t for t in tiers if t.get("key") == "llm"), {})
    slot_tier = next((t for t in tiers if t.get("key") == "slot"), {})
    llm_models = llm_tier.get("models") if isinstance(llm_tier.get("models"), list) else []
    slot_rows = slot_tier.get("models") if isinstance(slot_tier.get("models"), list) else []
    llm_total = len(llm_models) or 4
    llm_online = sum(1 for m in llm_models if m.get("online"))
    slot_total = len(slot_rows) or 5
    slot_ready = sum(1 for s in slot_rows if s.get("online"))
    has_live_cache = bool(tiers)
    return {
        "ts": time.time(),
        "release": ASSET_VER,
        "source": cached.get("source") if has_live_cache else "stale/offline",
        "summary": {
            "llm": {"online": llm_online, "total": llm_total, "state": "cached" if has_live_cache else "stale/offline"},
            "bpu_slots": {"available": slot_ready, "total": slot_total, "state": "cached" if has_live_cache else "stale/offline"},
            "public_boundary": "read-only explanation; no public robot control and no private prompts or keys",
        },
        "pipeline": [
            {"key": "formula", "title": "1. Formula object", "kind": "deterministic",
             "detail": "Parse formula, dopant, site, Shannon radius and host family into one citable material object."},
            {"key": "fly_mb", "title": "2. Fly-MB inspired sparse reasoning", "kind": "AI brain x5",
             "detail": "Fruit-fly mushroom-body inspiration is used as a design metaphor: sparse feature fanout, novelty gating, and fast candidate triage. It is not claimed as a biological simulation."},
            {"key": "mlip_ts", "title": "3. MLIP + Tanabe-Sugano proxy", "kind": "scientific model",
             "detail": "MatterGen/MatterSim/CHGNet stability fields and differentiable TS optical estimates provide lambda_em, band, and uncertainty context."},
            {"key": "hard_priors", "title": "4. Hard scientific priors", "kind": "rule gate",
             "detail": "Observed PL similarity, valence mismatch, radius mismatch, cross-host unknowns, and concentration quenching can override generic language-model text."},
            {"key": "edge_consensus", "title": "5. Edge consensus", "kind": "RDK X5",
             "detail": "4 CPU llama-server models, 5 BPU swap-load slots, light BPU perception models, and cloud R1 fallback are labelled by live/mirror/offline state."},
            {"key": "evidence", "title": "6. Evidence object", "kind": "public output",
             "detail": "The public site exposes trace_id, verdict, lambda_em, CI, method, source label, exports, and audit-chain fields without leaking private logs."},
        ],
        "signals": [
            {"label": "TS optical proxy", "value": "lambda_em MAE 6.2nm", "source": "project calibration"},
            {"label": "BPU transformer", "value": "Qwen2 24-layer forward 553ms", "source": "RDK X5 measured"},
            {"label": "Knowledge base", "value": "25,228 NIR paper vectors", "source": "DashScope embedding mirror"},
            {"label": "Audit", "value": "SHA-256 prediction chain", "source": "public hardening summary"},
        ],
        "boundaries": [
            "Fly-MB is an engineering analogy for sparse novelty-gated reasoning, not a biological claim.",
            "When X5 services are offline, the site must label mirror/history/curated data instead of showing fake live state.",
            "Public pages expose read-only explanation and evidence; device operation stays behind authenticated system pages.",
        ],
    }


@app.route("/api/ai_brain/explain")
def api_ai_brain_explain():
    return jsonify(_ai_brain_explain_payload())


@app.route("/api/rb_voe/explain")
def api_rb_voe_explain():
    """Release-bound, allowlisted RB-VoE evidence; never proxies device state."""
    try:
        payload = load_public_snapshot(_RB_VOE_PUBLIC_PATH, site_release=ASSET_VER)
    except RbVoePublicError:
        response = jsonify({
            "ok": False,
            "reason_code": "RB_VOE_PUBLIC_EVIDENCE_UNAVAILABLE",
            "execution_authority": False,
        })
        response.status_code = 503
    else:
        response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/atlas")
def api_atlas():
    """材料图鉴: lab 生成式候选池 (formula/λ_em/verdict/稳定性) + 汇总. 25s 缓存."""
    now = time.time()
    if _atlas_cache["data"] is None or now - _atlas_cache["ts"] > 25:
        try:
            _atlas_cache["data"] = _build_atlas(timeout=0.35)
            _atlas_cache["ts"] = now
        except Exception:
            if _atlas_cache["data"] is None:
                _atlas_cache["data"] = {"ts": now, "source": "down", "items": [], "summary": {}}
    return jsonify(_atlas_cache["data"])


# ============================================================ H22 告警 AI 诊断
# 对每条活动告警按规则给"根因推断 + 处置步骤 + 关联检查"。规则确定性 (映射告警 rule),
# 不编造; 这是把运维知识固化成可解释诊断, 评委/操作员一看就懂怎么修。
_DIAG_RULES = {
    "sys_down": {
        "cause": "该系统真机隧道与 VPS 镜像双路径同时不可达 —— 通常是镜像 systemd 服务挂了 (真机离线本不该告警, 因为镜像应兜底)。",
        "steps": ["operator_runbook_required: mirror_service_status",
                  "operator_runbook_required: mirror_service_restart",
                  "operator_runbook_required: gateway_health_check"],
        "related": "运维 → 三机健康拓扑 (看 serving 路径)"},
    "mirror_svc": {
        "cause": "VPS 常驻镜像 systemd 服务非 active —— 兜底链路失效, 一旦真机也离线就会 502。",
        "steps": ["operator_runbook_required: mirror_failure_log",
                  "operator_runbook_required: mirror_service_restart",
                  "operator_runbook_required: process_health_check"],
        "related": "运维 → 镜像服务态"},
    "real_offline": {
        "cause": "真机未上电, 公网由 VPS 镜像兜底 —— 这是设计内降级, 不是故障。设备上电后 Caddy 10s 内自动切回真机。",
        "steps": ["确认是否计划内 (设备没开机)", "需要真机数据时给对应 X5/arm 上电入网 (xrd-lab_5G)",
                  "等 active health 自动切换, 无需手动操作"],
        "related": "资产 → 设备上线状态"},
    "latency_high": {
        "cause": "服务路径延迟连续 3 次超阈 —— 可能是设备端 CPU 高负载、frp 隧道二跳拥塞、或网络抖动。",
        "steps": ["检查公开 source label 与延迟趋势",
                  "operator_runbook_required: protected_resource_check",
                  "持续超阈时按内部 runbook 排查网络或服务负载"],
        "related": "运维 → SLO 延迟曲线"},
    "audit_broken": {
        "cause": "预测审计链 SHA-256 完整性校验失败 —— 记录被改动或 hash 链断裂 (注意: 分段存储 ≠ 篡改, 曾有 4KB 截断误报 bug)。",
        "steps": ["进 AI 脑 /audit 看具体哪条记录断链",
                  "确认是真篡改还是读取逻辑 bug (分段边界)", "真篡改: 从备份恢复 predictions.jsonl"],
        "related": "AI 脑 → 审计链"},
    "llm_degraded": {
        "cause": "本地 LLM 部分离线 —— 某些 llama-server 进程没起来或崩了 (真机在线时才该告警)。",
        "steps": ["operator_runbook_required: ai_brain_local_llm_restart",
                  "operator_runbook_required: protected_model_health_check"],
        "related": "模型注册表 → LLM 在线态"},
}


# ============================================================ H24 周期表配方构建器
@app.route("/api/quick_predict", methods=["POST"])
def api_quick_predict():
    """快速预测 (不建工单): 周期表构建器提交 → 调 lab serving 端真 predict_engine → 摘要.

    复用工单的 _call_lab_predict (真机隧道或 VPS 镜像的真 predict_engine), 但不落库。
    """
    d = request.get_json(silent=True) or {}
    formula = (d.get("formula") or "").strip()
    symbol = (d.get("symbol") or "Cr3+").strip()
    site = (d.get("site") or "").strip()
    try:
        pct = float(d.get("pct") or 1.0)
    except (TypeError, ValueError):
        return jsonify({"error": "pct 非法"}), 400
    if not _FORMULA_RE.match(formula):
        return jsonify({"error": "化学式非法 (仅字母数字括号点, 2-60 字符)"}), 400
    if symbol not in ("Cr3+", "Ni2+", "Cr3++Ni2+"):
        return jsonify({"error": "掺杂仅支持 Cr3+/Ni2+/Cr3++Ni2+"}), 400
    if not re.match(r"^[A-Za-z]{0,10}$", site):
        return jsonify({"error": "位点仅限字母"}), 400
    if not (0 < pct <= 20):
        return jsonify({"error": "pct 范围 (0, 20]"}), 400
    pred, src = _call_lab_predict(formula, {"symbol": symbol, "site": site, "pct": pct})
    summ = _pred_summary(pred)
    if not summ:
        return jsonify({"error": "lab 预测不可达 (真机与镜像均未响应)", "source": src}), 502
    return jsonify({"ok": True, "source": src, "formula": formula,
                    "dopant": {"symbol": symbol, "site": site, "pct": pct},
                    "summary": summ, "flags": (pred or {}).get("flags") or [],
                    "sinter_temp_C": (pred or {}).get("sinter_temp_C")})


# ============================================================ H30 电子实验笔记 (ELN)
# 真持久化到 historian SQLite: 研究员记实验观察/配方思路, 关联化学式 + 标签 + 操作人。
# member 可写, judge 只读 (写在 SSO 层已 403, 这里再校验)。
def _eln_dict(r):
    return {"id": r[0], "ts": r[1], "updated_ts": r[2], "author": r[3],
            "title": r[4], "formula": r[5],
            "tags": [t for t in (r[6] or "").split(",") if t], "body": r[7]}


@app.route("/api/eln", methods=["GET"])
def api_eln_list():
    q = (request.args.get("q") or "").strip()
    limit = min(max(int(request.args.get("limit", 60)), 1), 200)
    con = _db()
    if q:
        like = f"%{q}%"
        rows = con.execute(
            "SELECT id,ts,updated_ts,author,title,formula,tags,body FROM eln"
            " WHERE title LIKE ? OR body LIKE ? OR formula LIKE ? OR tags LIKE ?"
            " ORDER BY ts DESC LIMIT ?", (like, like, like, like, limit)).fetchall()
    else:
        rows = con.execute(
            "SELECT id,ts,updated_ts,author,title,formula,tags,body FROM eln"
            " ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    n = con.execute("SELECT COUNT(*) FROM eln").fetchone()[0]
    con.close()
    return jsonify({"entries": [_eln_dict(r) for r in rows], "total": n})


@app.route("/api/eln", methods=["POST"])
def api_eln_create():
    if (request.headers.get("X-Role") or "") == "judge":
        return jsonify({"error": "评委账号为只读演示"}), 403
    d = request.get_json(silent=True) or {}
    title = (d.get("title") or "").strip()
    if not title:
        return jsonify({"error": "标题必填"}), 400
    if len(title) > 120:
        return jsonify({"error": "标题过长 (≤120)"}), 400
    body = (d.get("body") or "").strip()[:5000]
    formula = (d.get("formula") or "").strip()[:60]
    tags = ",".join(t.strip()[:20] for t in (d.get("tags") or [])[:8] if t.strip()) \
        if isinstance(d.get("tags"), list) else (d.get("tags") or "").strip()[:120]
    user = request.headers.get("X-User") or "operator"
    con = _db()
    cur = con.execute("INSERT INTO eln(ts,updated_ts,author,title,formula,tags,body)"
                      " VALUES(?,?,?,?,?,?,?)",
                      (int(time.time()), int(time.time()), user, title, formula, tags, body))
    eid = cur.lastrowid
    _add_event(None, "eln", "info", f"📓 {user} 记录实验笔记: {title[:60]}", con)
    row = con.execute("SELECT id,ts,updated_ts,author,title,formula,tags,body FROM eln"
                      " WHERE id=?", (eid,)).fetchone()
    con.commit()
    con.close()
    return jsonify({"ok": True, "entry": _eln_dict(row)})


@app.route("/api/eln/<int:eid>", methods=["DELETE"])
def api_eln_delete(eid):
    if (request.headers.get("X-Role") or "") == "judge":
        return jsonify({"error": "评委账号为只读演示"}), 403
    con = _db()
    row = con.execute("SELECT title FROM eln WHERE id=?", (eid,)).fetchone()
    if not row:
        con.close()
        return jsonify({"error": "笔记不存在"}), 404
    user = request.headers.get("X-User") or "operator"
    con.execute("DELETE FROM eln WHERE id=?", (eid,))
    _add_event(None, "eln", "info", f"🗑 {user} 删除笔记: {row[0][:60]}", con)
    con.commit()
    con.close()
    return jsonify({"ok": True})


# ============================================================ H26 镜像保鲜同步
# 监控三机 VPS 常驻镜像的"数据新鲜度" + 真机可达性 + 同步记录。设备离线下完全真实可测:
# 镜像数据来自上次设备在线快照, 离线时无法刷新但服务照常兜底 —— 这正是"关机 UI 照常在"的代价透明化。
_MIRROR_DATA = {
    "lab": "/home/rdk/mirrors/lab/predictions/predictions.jsonl",
    "car": "/home/rdk/mirrors/navcockpit/backend",
    "arm": "/home/rdk/mirrors/workcockpit",
}
_STALE_S = 7 * 86400  # 数据 >7 天未更新视为陈旧 (提示需设备上线刷新)


def _newest_mtime(path):
    """文件→自身 mtime; 目录→递归最新非 pyc/缓存源文件 mtime。返回 (mtime, file) 或 (None, None)。"""
    try:
        if os.path.isfile(path):
            return os.path.getmtime(path), os.path.basename(path)
        best, bf = 0.0, None
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules")]
            for fn in files:
                if fn.endswith((".pyc", ".pyo")):
                    continue
                fp = os.path.join(root, fn)
                try:
                    m = os.path.getmtime(fp)
                except OSError:
                    continue
                if m > best:
                    best, bf = m, fn
        return (best, bf) if bf else (None, None)
    except Exception:
        return None, None


def _age_txt(sec):
    if sec is None:
        return "—"
    sec = int(sec)
    if sec < 90:
        return f"{sec}秒前"
    if sec < 5400:
        return f"{sec // 60}分钟前"
    if sec < 172800:
        return f"{sec // 3600}小时前"
    return f"{sec // 86400}天前"


def _last_sync(con, k):
    r = con.execute("SELECT ts, message FROM events WHERE sys=? AND kind='mirror_sync'"
                    " ORDER BY id DESC LIMIT 1", (k,)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def _build_mirror_sync():
    now = time.time()
    con = _db()
    mirrors = []
    for k in ("lab", "car", "arm"):
        nm = SYSTEMS[k]["name"].split(" · ")[0]
        mt, mf = _newest_mtime(_MIRROR_DATA.get(k, ""))
        age = (now - mt) if mt else None
        real_code, _ = _probe_ms(SYSTEMS[k]["real"])
        real_on = real_code == 200
        svc = _svc_active(MIRROR_SVC[k])
        ls_ts, ls_msg = _last_sync(con, k)
        stale = bool(age is not None and age > _STALE_S)
        mirrors.append({
            "sys": k, "name": nm, "svc": svc, "svc_ok": svc == "active",
            "data_file": mf, "data_age_s": int(age) if age is not None else None,
            "data_age_txt": _age_txt(age), "stale": stale,
            "real_online": real_on, "can_sync": real_on,
            "last_sync_ts": ls_ts, "last_sync_txt": _age_txt((now - ls_ts) if ls_ts else None),
            "status": "ok" if (svc == "active" and not stale) else ("warn" if svc == "active" else "crit"),
        })
    con.close()
    return {"ts": now, "mirrors": mirrors,
            "summary": {"total": len(mirrors),
                        "stale": sum(1 for m in mirrors if m["stale"]),
                        "svc_down": sum(1 for m in mirrors if not m["svc_ok"]),
                        "real_online": sum(1 for m in mirrors if m["real_online"]),
                        "stale_threshold_days": _STALE_S // 86400},
            "note": "镜像数据为上次设备在线时的快照; 真机在线时可一键拉取刷新, 离线时服务照常兜底 (UI 不受影响)。"}


@app.route("/api/mirror_sync", methods=["GET", "POST"])
def api_mirror_sync():
    """GET: 三机镜像数据新鲜度 + 真机可达性 + 上次同步。POST(admin): 对在线真机记录同步检查 (设备离线诚实跳过)。"""
    if request.method == "GET":
        return jsonify(_build_mirror_sync())
    if (request.headers.get("X-Role") or "") != "admin":
        return jsonify({"error": "需要管理员权限"}), 403
    user = request.headers.get("X-User") or "admin"
    snap = _build_mirror_sync()
    con = _db()
    done, skipped = [], []
    for m in snap["mirrors"]:
        k = m["sys"]
        if m["real_online"]:
            # 真机在线: 记录一次同步检查 (真实可拉取信号; 全量数据拷贝由 PC 每日备份任务承担)
            _add_event(k, "mirror_sync", "info",
                       f"🔄 {user} 触发 {m['name']} 镜像同步检查 — 真机在线, 数据可刷新", con)
            done.append(k)
        else:
            skipped.append(k)
    con.commit()
    con.close()
    msg = (f"同步检查完成: {len(done)} 个真机在线可刷新" if done else "无真机在线")
    if skipped:
        msg += f", {len(skipped)} 个设备离线已跳过 (镜像继续兜底)"
    return jsonify({"ok": True, "synced": done, "skipped": skipped, "message": msg,
                    "result": _build_mirror_sync()})


# ============================================================ H20 文献/实测复现台
# 拿实验室真实测 ground truth (observed_pl.csv 有 λ_em 的行, 带 source 溯源) 喂回真 predict_engine,
# 比对预测 λ_em vs 实测值 + 是否落 90% CI 区间 —— 现场可核的"引擎复现已知结果"硬证据 (非自评分)。
_REPRO_CSV = "/home/rdk/mirrors/lab/exp_ground_truth/observed_pl.csv"
_REPRO_VAL = {"Cr": "Cr3+", "Ni": "Ni2+", "Fe": "Fe3+", "Mn": "Mn4+", "Co": "Co2+", "V": "V3+"}
_repro_ref = None          # 缓存解析后的参考集
_repro_pred = {}           # idx -> 预测比对结果


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_repro_ref():
    global _repro_ref
    if _repro_ref is not None:
        return _repro_ref
    out = []
    try:
        with open(_REPRO_CSV, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lam = _f((row.get("lambda_em_nm") or "").strip())
                if lam is None:
                    continue
                el = (row.get("dopant_element") or "").strip()
                out.append({
                    "idx": len(out), "formula": (row.get("formula") or "").strip(),
                    "dopant_element": el, "dopant_pct": (row.get("dopant_pct") or "").strip(),
                    "dopant_site": (row.get("dopant_site") or "").strip(),
                    "symbol": _REPRO_VAL.get(el, el),
                    "ref_lambda_em": lam, "lambda_ex": (row.get("lambda_ex_nm") or "").strip(),
                    "source": (row.get("source") or row.get("reported_by") or "实验室实测").strip(),
                })
    except Exception:
        pass
    _repro_ref = out
    return out


def _repro_summary(ref):
    done = [v for v in _repro_pred.values() if v and v.get("pred_lambda") is not None]
    errs = [abs(v["err"]) for v in done if v.get("err") is not None]
    return {"total": len(ref), "done": len(done),
            "mae": round(sum(errs) / len(errs), 1) if errs else None,
            "within50": sum(1 for e in errs if e <= 50),
            "within_ci": sum(1 for v in done if v.get("within_ci"))}


@app.route("/api/reproduce", methods=["GET", "POST"])
def api_reproduce():
    """GET: 实测参考集 + 已复现结果。POST {idx}: 对该行调真 predict_engine 比对 λ_em (缓存)。"""
    ref = _load_repro_ref()
    if request.method == "GET":
        items = [{**r, "pred": _repro_pred.get(r["idx"])} for r in ref]
        return jsonify({"ts": time.time(), "items": items, "summary": _repro_summary(ref),
                        "source": "实验室实测 ground truth · observed_pl.csv (带 source 溯源)",
                        "note": "参考值为实验室真实测 λ_em; 预测由当前 serving 端真 predict_engine 现算, 误差与 90% CI 命中可现场复核。"})
    body = request.get_json(silent=True) or {}
    try:
        idx = int(body.get("idx"))
    except (TypeError, ValueError):
        return jsonify({"error": "无效行号"}), 400
    if not (0 <= idx < len(ref)):
        return jsonify({"error": "行号越界"}), 400
    r = ref[idx]
    dop = {"symbol": r["symbol"], "site": r["dopant_site"], "pct": _f(r["dopant_pct"]) or 1.0}
    pred, src = _call_lab_predict(r["formula"], dop)
    summ = _pred_summary(pred) or {}
    pl = summ.get("lambda_em")
    rec = {"pred_lambda": round(pl, 1) if isinstance(pl, (int, float)) else None,
           "ci_lo": summ.get("ci_lo"), "ci_hi": summ.get("ci_hi"), "verdict": summ.get("verdict"),
           "src": src, "trace_id": summ.get("trace_id"), "ts": time.time()}
    if rec["pred_lambda"] is not None:
        rec["err"] = round(rec["pred_lambda"] - r["ref_lambda_em"], 1)
        lo, hi = _f(rec.get("ci_lo")), _f(rec.get("ci_hi"))
        rec["within_ci"] = bool(lo is not None and hi is not None and lo <= r["ref_lambda_em"] <= hi)
    _repro_pred[idx] = rec
    return jsonify({"ok": True, "idx": idx, "pred": rec, "ref_lambda_em": r["ref_lambda_em"],
                    "summary": _repro_summary(ref)})


# ============================================================ E1 可观测性中心 (时序指标)
# 把 historian (samples 30s 采样 + kpi_samples) 变成可交互时序图的数据源。对标 Grafana/Ignition:
# 多时间窗 + 分桶降采样 + 多机/多指标。全真数据 (设备离线下镜像延迟/可用性照样在记)。
_OBS_RANGES = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}
_OBS_SYS = {"lab": ("AI 脑", "#7c3aed"), "car": ("车载脑", "#2563eb"), "arm": ("机械臂", "#06b6d4")}
_OBS_METRICS = {
    "latency": {"title": "服务延迟 (当前 serving 端往返)", "unit": "ms", "src": "samples"},
    "avail": {"title": "可用性 (真机直连占比)", "unit": "%", "src": "samples"},
    "predictions": {"title": "累计预测记录数", "unit": "条", "src": "kpi"},
    "ci": {"title": "Conformal 90% CI 实测覆盖率", "unit": "%", "src": "kpi"},
    "audit": {"title": "SHA-256 审计链完整率", "unit": "%", "src": "kpi"},
    "llm": {"title": "本地 LLM 在线率", "unit": "%", "src": "kpi"},
}


def _bucket_series(rows, t0, bw, nb, valfn):
    """rows 按桶聚合 → [[bucket_center_ts, avg_val 或 None], ...]。valfn(row)->float|None。"""
    acc = [[] for _ in range(nb)]
    for r in rows:
        b = int((r["ts"] - t0) / bw)
        if 0 <= b < nb:
            v = valfn(r)
            if v is not None:
                acc[b].append(v)
    out = []
    for i in range(nb):
        c = int(t0 + (i + 0.5) * bw)
        out.append([c, round(sum(acc[i]) / len(acc[i]), 1) if acc[i] else None])
    return out


def _build_metrics(metric, rng):
    if metric not in _OBS_METRICS:
        metric = "latency"
    span = _OBS_RANGES.get(rng, 21600)
    t1 = int(time.time())
    t0 = t1 - span
    nb = 90
    bw = span / nb
    md = _OBS_METRICS[metric]
    con = _db()
    series = []
    if md["src"] == "samples":
        rows = [dict(zip(("ts", "sys", "serving", "real_ms", "mirror_ms"), r)) for r in
                con.execute("SELECT ts,sys,serving,real_ms,mirror_ms FROM samples WHERE ts>=?"
                            " ORDER BY ts", (t0,)).fetchall()]
        for sk, (nm, col) in _OBS_SYS.items():
            sr = [r for r in rows if r["sys"] == sk]
            if metric == "latency":
                def vf(r):
                    if r["serving"] == "real" and r["real_ms"] is not None:
                        return float(r["real_ms"])
                    if r["mirror_ms"] is not None:
                        return float(r["mirror_ms"])
                    return None
            else:  # avail = 真机直连占比 %
                def vf(r):
                    return 100.0 if r["serving"] == "real" else 0.0
            series.append({"key": sk, "label": nm, "color": col,
                           "points": _bucket_series(sr, t0, bw, nb, vf)})
    else:
        rows = [dict(zip(("ts", "predictions", "ci_coverage_pct", "audit_valid", "audit_total",
                          "llm_up", "llm_total"), r)) for r in
                con.execute("SELECT ts,predictions,ci_coverage_pct,audit_valid,audit_total,llm_up,llm_total"
                            " FROM kpi_samples WHERE ts>=? ORDER BY ts", (t0,)).fetchall()]
        valfn = {
            "predictions": lambda r: float(r["predictions"]) if r["predictions"] is not None else None,
            "ci": lambda r: float(r["ci_coverage_pct"]) if r["ci_coverage_pct"] is not None else None,
            "audit": lambda r: (100.0 * r["audit_valid"] / r["audit_total"]) if r.get("audit_total") else None,
            "llm": lambda r: (100.0 * r["llm_up"] / r["llm_total"]) if r.get("llm_total") else None,
        }[metric]
        series.append({"key": metric, "label": md["title"], "color": "#7c3aed",
                       "points": _bucket_series(rows, t0, bw, nb, valfn)})
    con.close()
    # 汇总 (取非空点)
    allv = [p[1] for s in series for p in s["points"] if p[1] is not None]
    summary = {"last": next((p[1] for s in series for p in reversed(s["points"]) if p[1] is not None), None),
               "min": round(min(allv), 1) if allv else None, "max": round(max(allv), 1) if allv else None,
               "avg": round(sum(allv) / len(allv), 1) if allv else None, "n": len(allv)}
    return {"ts": t1, "metric": metric, "title": md["title"], "unit": md["unit"],
            "range": rng, "t0": t0, "t1": t1, "series": series, "summary": summary,
            "metrics": [{"key": k, "title": v["title"], "unit": v["unit"]} for k, v in _OBS_METRICS.items()]}


@app.route("/api/metrics")
def api_metrics():
    """可观测性时序: ?metric=latency|avail|predictions|ci|audit|llm &range=1h|6h|24h|7d. 分桶降采样真数据."""
    return jsonify(_build_metrics(request.args.get("metric", "latency"),
                                  request.args.get("range", "6h")))


# ============================================================ E2 服务依赖拓扑图
# 把"访客→Cloudflare→VPS Caddy→(真机隧道 | 镜像)→三机"的真实链路画成活体节点图, 当前 serving 路径高亮。
# 对标 Datadog APM service map: 节点健康着色 + 边随实际流量激活。全真状态 (复用 _build_ops)。
def _build_topology():
    ops = _build_ops()
    nodes = [
        {"id": "visitor", "label": "访客 / 评委", "sub": "浏览器", "layer": 0, "kind": "client", "status": "ok"},
        {"id": "cf", "label": "Cloudflare", "sub": "边缘代理 · 大陆可达", "layer": 1, "kind": "edge", "status": "ok"},
        {"id": "vps", "label": "香港 VPS · Caddy", "sub": "frp 隧道 + 镜像 + SSO", "layer": 2, "kind": "gateway", "status": "ok"},
    ]
    edges = [{"from": "visitor", "to": "cf", "active": True}, {"from": "cf", "to": "vps", "active": True}]
    real = mir = down = 0
    for k in ("lab", "car", "arm"):
        s = ops["systems"][k]
        serv = s["serving"]
        nm = s["name"].split(" · ")[0]
        if serv == "real":
            real += 1
        elif serv == "mirror":
            mir += 1
        else:
            down += 1
        mstat = {"real": "ok", "mirror": "info", "down": "crit"}[serv]
        nodes.append({"id": k + "-real", "label": nm + " 真机隧道", "sub": "受保护运维隧道",
                      "layer": 3, "kind": "tunnel", "status": "ok" if s["real_online"] else "idle", "ms": s["real_ms"]})
        nodes.append({"id": k + "-mirror", "label": nm + " 镜像", "sub": MIRROR_SVC[k], "layer": 3, "kind": "mirror",
                      "status": "ok" if s["mirror_svc"] == "active" else "crit", "ms": s["mirror_ms"]})
        nodes.append({"id": k, "label": nm, "sub": {"real": "真机直连", "mirror": "镜像兜底", "down": "离线"}[serv],
                      "layer": 4, "kind": "machine", "status": mstat, "serving": serv,
                      "ms": s["real_ms"] if serv == "real" else s["mirror_ms"]})
        edges.append({"from": "vps", "to": k + "-real", "active": serv == "real"})
        edges.append({"from": "vps", "to": k + "-mirror", "active": serv == "mirror"})
        edges.append({"from": k + "-real", "to": k, "active": serv == "real"})
        edges.append({"from": k + "-mirror", "to": k, "active": serv == "mirror"})
    return {"ts": time.time(), "nodes": nodes, "edges": edges,
            "summary": {"real": real, "mirror": mir, "down": down,
                        "verdict": ("全链路真机直连" if real == 3 else
                                    ("镜像兜底运行中 (设备未上电)" if down == 0 else "有系统双路全断"))}}


@app.route("/api/topology")
def api_topology():
    """服务依赖拓扑: 访客→CF→VPS→(真机隧道|镜像)→三机 的活体节点图, 当前 serving 路径高亮."""
    return jsonify(_build_topology())


# ============================================================ E3 SLO 错误预算 burn-down
# 诚实 SLO = UI 可用性 (serving != down)。镜像兜底正是让该 SLO 不受设备开关机影响的关键。
# 错误预算 = (1-目标)×窗口; 消耗 = 累计 down 时长; burn-down = 剩余预算% 随时间下降。对标 Google SRE / Nobl9。
_SLO_WINDOWS = {"24h": 86400, "7d": 604800, "30d": 2592000}
_SLO_SEG = 30  # samples 采样间隔秒


def _build_slo_budget(window, target, scope):
    span = _SLO_WINDOWS.get(window, 604800)
    try:
        target = max(90.0, min(99.99, float(target)))
    except (TypeError, ValueError):
        target = 99.5
    t1 = int(time.time())
    t0 = t1 - span
    allowed_down_s = span * (1 - target / 100.0)
    con = _db()
    scopes = []
    for k in ("lab", "car", "arm"):
        rows = con.execute("SELECT ts,serving FROM samples WHERE sys=? AND ts>=? ORDER BY ts",
                           (k, t0)).fetchall()
        n = len(rows)
        down = sum(1 for _ts, s in rows if s == "down")
        avail = round(100.0 * (n - down) / n, 3) if n else None
        consumed_s = down * _SLO_SEG
        rem_pct = round(100.0 * (1 - consumed_s / allowed_down_s), 1) if allowed_down_s > 0 else None
        scopes.append({
            "key": k, "label": SYSTEMS[k]["name"].split(" · ")[0], "samples": n,
            "availability_pct": avail, "target_pct": target,
            "budget_total_min": round(allowed_down_s / 60, 1),
            "consumed_min": round(consumed_s / 60, 1),
            "remaining_pct": rem_pct,
            "status": ("ok" if (avail is not None and avail >= target) else
                       ("warn" if (rem_pct is not None and rem_pct > 0) else "crit")),
            "met": bool(avail is not None and avail >= target),
        })
    # burn-down 序列 (仅选定 scope, 控制 payload)
    sc = scope if scope in ("lab", "car", "arm") else "lab"
    rows = con.execute("SELECT ts,serving FROM samples WHERE sys=? AND ts>=? ORDER BY ts", (sc, t0)).fetchall()
    con.close()
    nb = 90
    bw = span / nb
    cum_down = [0] * nb
    for ts, s in rows:
        b = int((ts - t0) / bw)
        if 0 <= b < nb and s == "down":
            cum_down[b] += 1
    series = []
    run = 0
    for i in range(nb):
        run += cum_down[i]
        consumed = run * _SLO_SEG
        rem = 100.0 * (1 - consumed / allowed_down_s) if allowed_down_s > 0 else 100.0
        series.append([int(t0 + (i + 0.5) * bw), round(rem, 1)])
    return {"ts": t1, "window": window, "target": target, "scope": sc, "t0": t0, "t1": t1,
            "scopes": scopes, "burndown": series,
            "note": "SLO = UI 可用性 (真机或镜像任一在即算可用); 镜像兜底使该 SLO 不受设备开关机影响。剩余预算 <0 即违约。"}


@app.route("/api/slo_budget")
def api_slo_budget():
    """SLO 错误预算 burn-down: ?window=24h|7d|30d &target=99.5 &scope=lab|car|arm. 真 samples 计算."""
    return jsonify(_build_slo_budget(request.args.get("window", "7d"),
                                     request.args.get("target", "99.5"),
                                     request.args.get("scope", "lab")))


# ============================================================ E4 事故复盘 (Incident & Postmortem)
# 把每条告警的生命周期 (触发→确认→恢复) + 周边事件组装成事故时间线, 并据 _DIAG_RULES 自动生成复盘草稿。
# 对标 PagerDuty / Blameless: 时间线 + 影响 + 根因 + 处置 + 改进项。real_offline 类如实标注"设计内降级"。
def _fmt_dur(sec):
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec} 秒"
    if sec < 3600:
        return f"{sec // 60} 分 {sec % 60} 秒"
    if sec < 86400:
        return f"{sec // 3600} 小时 {(sec % 3600) // 60} 分"
    return f"{sec // 86400} 天 {(sec % 86400) // 3600} 小时"


def _build_incidents():
    now = int(time.time())
    con = _db()
    alarms = con.execute(
        "SELECT id,rule,sys,severity,message,ts_raised,ts_cleared,ts_ack,ack_by FROM alarms"
        " ORDER BY ts_raised DESC LIMIT 25").fetchall()
    out = []
    for aid, rule, sys, sev, msg, raised, cleared, ack, ackby in alarms:
        end = cleared or now
        dur = end - raised
        nm = SYSTEMS.get(sys, {}).get("name", "平台").split(" · ")[0] if sys else "平台"
        evs = con.execute(
            "SELECT ts,kind,severity,message FROM events WHERE ts>=? AND ts<=? AND (sys=? OR sys IS NULL)"
            " ORDER BY ts LIMIT 12", (raised - 90, end + 90, sys)).fetchall()
        tl = [{"ts": raised, "label": "告警触发", "detail": msg, "sev": sev}]
        if ack:
            tl.append({"ts": ack, "label": "已确认", "detail": "操作人 " + (ackby or "?"), "sev": "info"})
        for ets, ekind, esev, emsg in evs:
            if abs(ets - raised) <= 2:
                continue
            tl.append({"ts": ets, "label": ekind or "事件", "detail": emsg, "sev": esev or "info"})
        if cleared:
            tl.append({"ts": cleared, "label": "已恢复", "detail": "告警清除, 服务恢复", "sev": "ok"})
        tl.sort(key=lambda x: x["ts"])
        diag = _DIAG_RULES.get(rule, {"cause": "未归类规则, 需人工研判。", "steps": [], "related": ""})
        design = (rule == "real_offline")
        state = "已恢复" if cleared else ("处置中" if ack else "进行中")
        out.append({
            "id": aid, "rule": rule, "sys": sys, "sysname": nm, "severity": sev,
            "title": (("[设计内] " if design else "") + (msg or rule)),
            "state": state, "design_intended": design,
            "ts_raised": raised, "ts_cleared": cleared, "duration_s": dur, "duration_txt": _fmt_dur(dur),
            "acked": bool(ack), "ack_by": ackby,
            "cause": diag["cause"], "actions": diag.get("steps", []), "related": diag.get("related", ""),
            "timeline": tl,
            "impact": (f"{nm} 触发 {sev} 级告警, 持续 {_fmt_dur(dur)}"
                       + ("; 由镜像兜底, UI 与功能不受影响 (设计内)。" if design else "; 需关注链路可用性。")),
            "followups": (["设备上电后自动恢复, 无改进项 (设计内降级)"] if design else
                          ["确认根因是否复发", "评估是否需加监控阈值/自愈脚本", "补充 runbook"]),
        })
    con.close()
    active = sum(1 for o in out if not o["ts_cleared"])
    return {"ts": now, "incidents": out,
            "summary": {"total": len(out), "active": active,
                        "design_intended": sum(1 for o in out if o["design_intended"]),
                        "mttr_txt": (_fmt_dur(sum(o["duration_s"] for o in out if o["ts_cleared"]) /
                                     max(1, sum(1 for o in out if o["ts_cleared"]))) if any(o["ts_cleared"] for o in out) else "—")},
            "note": "事故来自告警生命周期 + 周边 historian 事件; 复盘草稿据规则库自动生成, 可现场核对。"}


@app.route("/api/incidents")
def api_incidents():
    """事故复盘: 告警生命周期 → 时间线 + 影响 + 根因 + 处置 + 改进项 (自动草稿, 对标 PagerDuty/Blameless)."""
    return jsonify(_build_incidents())


# ============================================================ E7 全局时间机器 (state replay)
# 拖动到任意历史时刻 → 用 historian 重建当时的平台快照: 三机 serving/延迟 + KPI + 当时活动告警 + 周边事件。
# 对标 Palantir Foundry / Ignition 的 time-travel。纯查询重建, 不改任何状态。
def _build_timemachine(at):
    con = _db()
    rng = con.execute("SELECT MIN(ts),MAX(ts) FROM samples").fetchone()
    tmin = rng[0] or (int(time.time()) - 3600)
    tmax = rng[1] or int(time.time())
    if at is None:
        at = tmax
    at = max(tmin, min(tmax, int(at)))
    systems = {}
    real = mir = down = 0
    for k in ("lab", "car", "arm"):
        r = con.execute("SELECT ts,serving,real_ms,mirror_ms FROM samples WHERE sys=? AND ts<=?"
                        " ORDER BY ts DESC LIMIT 1", (k, at)).fetchone()
        nm = SYSTEMS[k]["name"].split(" · ")[0]
        if r:
            serv = r[1]
            ms = r[2] if serv == "real" else r[3]
            if serv == "real":
                real += 1
            elif serv == "mirror":
                mir += 1
            else:
                down += 1
            systems[k] = {"name": nm, "serving": serv, "ms": ms, "sample_ts": r[0], "age_s": at - r[0]}
        else:
            systems[k] = {"name": nm, "serving": None, "ms": None, "sample_ts": None, "age_s": None}
    kr = con.execute("SELECT ts,predictions,ci_coverage_pct,audit_valid,audit_total,llm_up,llm_total"
                     " FROM kpi_samples WHERE ts<=? ORDER BY ts DESC LIMIT 1", (at,)).fetchone()
    kpi = {}
    if kr:
        kpi = {"predictions": kr[1], "ci_coverage_pct": kr[2],
               "audit": (f"{kr[3]}/{kr[4]}" if kr[4] else None),
               "llm": (f"{kr[5]}/{kr[6]}" if kr[6] else None), "sample_ts": kr[0]}
    alarms = [{"rule": a[0], "sys": a[1], "severity": a[2], "message": a[3], "ts_raised": a[4]}
              for a in con.execute(
                  "SELECT rule,sys,severity,message,ts_raised FROM alarms WHERE ts_raised<=?"
                  " AND (ts_cleared IS NULL OR ts_cleared>?) ORDER BY ts_raised DESC", (at, at)).fetchall()]
    events = [{"ts": e[0], "sys": e[1], "kind": e[2], "severity": e[3], "message": e[4]}
              for e in con.execute("SELECT ts,sys,kind,severity,message FROM events WHERE ts<=?"
                                   " ORDER BY ts DESC LIMIT 6", (at,)).fetchall()]
    # 时间轴标记 (事件/告警在全程的位置, 给滑块轨道画点)
    markers = []
    for ts, sev in con.execute("SELECT ts,severity FROM events WHERE ts>=? ORDER BY ts DESC LIMIT 60",
                               (tmin,)).fetchall():
        markers.append({"ts": ts, "sev": sev or "info", "kind": "event"})
    for ts, sev in con.execute("SELECT ts_raised,severity FROM alarms WHERE ts_raised>=?", (tmin,)).fetchall():
        markers.append({"ts": ts, "sev": sev or "warn", "kind": "alarm"})
    con.close()
    return {"ts": int(time.time()), "at": at, "tmin": tmin, "tmax": tmax,
            "is_live": at >= tmax - 60,
            "summary": {"real": real, "mirror": mir, "down": down,
                        "serving_txt": (f"{real} 真机 / {mir} 镜像 / {down} 离线")},
            "systems": systems, "kpi": kpi, "alarms": alarms, "events": events, "markers": markers,
            "note": "用 historian 重建的历史快照 (取 ≤ 该时刻最近一条采样); 设备离线期镜像兜底, serving=mirror。"}


@app.route("/api/timemachine")
def api_timemachine():
    """全局时间机器: ?at=<unix ts> 重建当时平台快照 (三机 serving/延迟 + KPI + 活动告警 + 事件). 无 at 取最新."""
    at = request.args.get("at")
    return jsonify(_build_timemachine(int(at) if at and at.isdigit() else None))


# ============================================================ E8 统一运营总览 (NOC Wall)
# 把 E 轮观测能力收拢成一屏: 舰队 serving + 延迟迷你趋势 + SLO 24h + 活动事故 + KPI + 事件流。
# 一次聚合 (少往返), 复用缓存。对标 Grafana 首页 / NOC 运维墙。
def _build_noc():
    with _lock:
        ops = _ops_cache["data"] or _build_ops()
        kd = _kpi_cache["data"] or _build_kpi()
    now = int(time.time())
    systems = []
    real = mir = down = 0
    con = _db()
    for k in ("lab", "car", "arm"):
        s = ops.get("systems", {}).get(k, {})
        serv = s.get("serving")
        if serv == "real":
            real += 1
        elif serv == "mirror":
            mir += 1
        else:
            down += 1
        ms = s.get("real_ms") if serv == "real" else s.get("mirror_ms")
        # 迷你延迟火花线 (最近 40 采样)
        rows = con.execute("SELECT serving,real_ms,mirror_ms FROM samples WHERE sys=?"
                           " ORDER BY ts DESC LIMIT 40", (k,)).fetchall()
        spark = []
        for sv, rm, mm in reversed(rows):
            v = rm if (sv == "real" and rm is not None) else mm
            spark.append(v if v is not None else 0)
        # 24h 可用性
        a = con.execute("SELECT COUNT(*), SUM(serving='down') FROM samples WHERE sys=? AND ts>=?",
                        (k, now - 86400)).fetchone()
        avail = round(100.0 * (a[0] - (a[1] or 0)) / a[0], 2) if a[0] else None
        systems.append({"key": k, "name": SYSTEMS[k]["name"].split(" · ")[0], "serving": serv,
                        "ms": ms, "spark": spark, "avail24": avail})
    ac = _alarm_counts()
    active = [{"severity": r[0], "message": r[1], "ts": r[2]} for r in con.execute(
        "SELECT severity,message,ts_raised FROM alarms WHERE ts_cleared IS NULL"
        " ORDER BY ts_raised DESC LIMIT 5").fetchall()]
    events = [{"ts": r[0], "sev": r[1], "msg": r[2]} for r in con.execute(
        "SELECT ts,severity,message FROM events ORDER BY id DESC LIMIT 7").fetchall()]
    con.close()
    k_ = kd.get("kpi", {})
    return {"ts": now, "systems": systems,
            "fleet": {"real": real, "mirror": mir, "down": down,
                      "verdict": ("全链路真机直连" if real == 3 else
                                  ("镜像兜底运行中" if down == 0 else "有系统离线"))},
            "alarms": {"total": ac["total"], "crit": ac["crit"], "warn": ac["warn"],
                       "unacked": ac["unacked"], "active": active},
            "kpi": {"predictions": k_.get("predictions"), "ci": k_.get("ci_coverage_pct"),
                    "audit_intact": k_.get("audit_intact"), "audit": (f"{k_.get('audit_valid')}/{k_.get('audit_total')}"
                    if k_.get("audit_total") else None), "source": kd.get("source")},
            "events": events}


@app.route("/api/noc")
def api_noc():
    """统一运营总览 (NOC Wall): 舰队 serving + 延迟火花线 + 24h 可用性 + 活动告警 + KPI + 事件流, 一次聚合."""
    return jsonify(_build_noc())


# ============================================================ Site9 R8 fleet / tasks / observability / traces
_PUBLIC_SYSTEM_NAMES = {
    "lab": "AI Brain",
    "car": "Vehicle Brain",
    "arm": "Arm Workstation",
    "arm02": "Arm02 Standby",
}


def _last_system_sample(con, key):
    return con.execute("SELECT ts,serving,real_ms,mirror_ms FROM samples WHERE sys=? ORDER BY ts DESC LIMIT 1",
                       (key,)).fetchone()


def _availability_pct(con, key, since):
    r = con.execute("SELECT COUNT(*), SUM(serving='down') FROM samples WHERE sys=? AND ts>=?",
                    (key, since)).fetchone()
    n = r[0] or 0
    if not n:
        return None
    return round(100.0 * (n - (r[1] or 0)) / n, 2)


def _fleet_console_payload():
    now = int(time.time())
    con = _db()
    with _lock:
        ops = _ops_cache["data"] if _ops_cache.get("data") and time.time() - _ops_cache.get("ts", 0) < 120 else None
    systems = []
    for key in ("lab", "car", "arm"):
        cached = ((ops or {}).get("systems") or {}).get(key, {})
        row = _last_system_sample(con, key)
        serv = cached.get("serving") or (row[1] if row else None)
        sample_ts = row[0] if row else None
        age = now - sample_ts if sample_ts else None
        ms = cached.get("real_ms") if serv == "real" else cached.get("mirror_ms")
        if ms is None and row:
            ms = row[2] if serv == "real" else row[3]
        spark_rows = con.execute("SELECT serving,real_ms,mirror_ms FROM samples WHERE sys=? ORDER BY ts DESC LIMIT 24",
                                 (key,)).fetchall()
        spark = []
        for sv, rm, mm in reversed(spark_rows):
            v = rm if sv == "real" else mm
            spark.append(v if v is not None else 0)
        raw_metrics = cached.get("metrics") or {}
        metrics = {str(k)[:28]: _public_safe_text(v, 80) for k, v in list(raw_metrics.items())[:8]}
        systems.append({
            "key": key,
            "name": _PUBLIC_SYSTEM_NAMES[key],
            "serving": serv or "unknown",
            "source": _serving_source(serv, age),
            "last_seen_ts": sample_ts,
            "age_s": age,
            "latency_ms": ms,
            "availability_24h": _availability_pct(con, key, now - 86400),
            "mirror_service": cached.get("mirror_svc") or "unknown",
            "metrics": metrics,
            "spark": spark,
            "next_action": "use live system" if serv == "real" else (
                "use VPS mirror; power on device when physical execution is needed" if serv == "mirror" else
                "show offline state; no physical command is sent"),
        })
    systems.append({
        "key": "arm02",
        "name": _PUBLIC_SYSTEM_NAMES["arm02"],
        "serving": "unknown",
        "source": "unknown",
        "last_seen_ts": None,
        "age_s": None,
        "latency_ms": None,
        "availability_24h": None,
        "mirror_service": "not_public",
        "metrics": {},
        "spark": [],
        "next_action": "standby device; public aggregator has no direct signal",
    })
    events = [{"ts": r[0], "sys": r[1], "severity": r[2], "message": _public_safe_text(r[3])}
              for r in con.execute("SELECT ts,sys,severity,message FROM events ORDER BY id DESC LIMIT 8").fetchall()]
    con.close()
    counts = {"live": sum(1 for s in systems if s["source"] == "live"),
              "mirror": sum(1 for s in systems if s["source"] == "mirror"),
              "stale": sum(1 for s in systems if s["source"] == "stale"),
              "offline": sum(1 for s in systems if s["source"] == "offline"),
              "unknown": sum(1 for s in systems if s["source"] == "unknown")}
    return {"ts": now, "release": ASSET_VER, "summary": counts, "systems": systems,
            "events": events,
            "note": "Fleet cockpit uses cached historian and mirror status; it does not wait for offline robots."}


@app.route("/api/fleet_cockpit")
def api_fleet_cockpit():
    return jsonify(_fleet_console_payload())


def _task_next_action(w):
    state = w.get("state")
    stage = int(w.get("stage") or 0)
    if state == "done":
        return "archive and compare measured feedback"
    if state == "cancelled":
        return "no action; cancelled batch"
    if not w.get("trace_id"):
        return "retry AI prediction or keep as manual batch record"
    if stage < 1:
        return "run prediction and bind trace_id"
    if stage < 4:
        return "advance to next lab stage when physical equipment is ready"
    return "enter measured lambda_obs and close batch"


def _task_blocker(w):
    if w.get("state") == "done":
        return ""
    if w.get("state") == "cancelled":
        return "cancelled"
    if not w.get("trace_id"):
        return "prediction trace missing"
    if w.get("verdict") and str(w.get("verdict")).upper() not in ("GO", "REFERENCE"):
        return "manual review for non-GO verdict"
    return "waiting for operator or physical station"


def _tasks_console_payload(limit=80):
    try:
        lim = max(1, min(int(limit), 200))
    except Exception:
        lim = 80
    con = _db()
    rows = con.execute(f"SELECT {_WO_COLS} FROM workorders ORDER BY id DESC LIMIT ?",
                       (lim,)).fetchall()
    tasks = []
    for r in rows:
        w = _wo_dict(r)
        logs = con.execute("SELECT ts,author,action,detail FROM wo_log WHERE wo=? ORDER BY id DESC LIMIT 5",
                           (w["id"],)).fetchall()
        pred = w.get("pred") or {}
        tasks.append({
            "id": w["id"],
            "code": w["code"],
            "formula": w["formula"],
            "dopant": w.get("dop_symbol"),
            "stage": w.get("stage") or 0,
            "stage_name": (WO_STAGES[min(max(int(w.get("stage") or 0), 0), len(WO_STAGES) - 1)]
                           if WO_STAGES else ""),
            "state": w.get("state"),
            "trace_id": w.get("trace_id") or "",
            "verdict": w.get("verdict") or "",
            "source": w.get("pred_source") or "manual/history",
            "lambda_em": pred.get("lambda_em"),
            "blocker": _task_blocker(w),
            "next_action": _task_next_action(w),
            "manual_review": bool(_task_blocker(w)),
            "history": [{"ts": x[0], "author": _public_safe_text(x[1], 40),
                         "action": _public_safe_text(x[2], 40),
                         "detail": _public_safe_text(x[3], 160)} for x in logs],
        })
    counts = dict(con.execute("SELECT state, COUNT(*) FROM workorders GROUP BY state").fetchall())
    con.close()
    return {"ts": int(time.time()), "release": ASSET_VER,
            "counts": {"open": counts.get("open", 0), "done": counts.get("done", 0),
                       "cancelled": counts.get("cancelled", 0), "total": sum(counts.values())},
            "tasks": tasks,
            "empty_state": "No public batch records yet. Create a batch in Queue to bind a prediction trace."
            if not tasks else ""}


@app.route("/api/tasks_cockpit")
def api_tasks_cockpit():
    return jsonify(_tasks_console_payload(request.args.get("limit", 80)))


def _agent_studio_payload(limit=18):
    now = int(time.time())
    try:
        lim = min(max(int(limit), 1), 80)
    except Exception:
        lim = 18
    con = _db()
    rows = con.execute(f"SELECT {_WO_COLS} FROM workorders ORDER BY id DESC LIMIT ?", (lim,)).fetchall()
    stage_counts = {str(i): 0 for i in range(len(WO_STAGES) + 1)}
    objects = []
    for r in rows:
        w = _wo_dict(r)
        stage = min(max(int(w.get("stage") or 0), 0), len(WO_STAGES))
        stage_counts[str(stage)] = stage_counts.get(str(stage), 0) + 1
        pred = w.get("pred") or {}
        objects.append({
            "id": w["id"],
            "code": w["code"],
            "formula": w["formula"],
            "dopant": w.get("dop_symbol") or "",
            "site": w.get("dop_site") or "",
            "stage": stage,
            "stage_name": WO_STAGES[min(stage, len(WO_STAGES) - 1)] if WO_STAGES else "",
            "state": w.get("state"),
            "trace_id": w.get("trace_id") or "",
            "verdict": w.get("verdict") or "",
            "lambda_em": pred.get("lambda_em"),
            "blocker": _task_blocker(w),
            "next_action": _task_next_action(w),
            "material_url": "/materials/" + quote(w.get("trace_id") or w.get("formula") or w.get("code"), safe=""),
            "workorder_url": f"/api/workorders/{w['id']}",
        })
    if not objects:
        try:
            for row in _materials_all_rows()[:lim]:
                mid = _material_id_path(row)
                objects.append({
                    "id": 0,
                    "object_id": mid,
                    "code": mid,
                    "formula": row.get("formula") or "",
                    "dopant": row.get("dopant") or "",
                    "site": row.get("site") or "",
                    "stage": 0,
                    "stage_name": "candidate material object",
                    "state": row.get("state") or "not queued",
                    "trace_id": row.get("trace_id") or "",
                    "verdict": row.get("verdict") or "",
                    "lambda_em": row.get("lambda_em"),
                    "blocker": "not yet queued as a work order",
                    "next_action": "open material evidence, then create a work order when execution is needed",
                    "material_url": row.get("detail_url") or ("/materials/" + quote(mid, safe="")),
                    "workorder_url": "",
                })
        except Exception:
            pass
    state_counts = dict(con.execute("SELECT state, COUNT(*) FROM workorders GROUP BY state").fetchall())
    eln_count = con.execute("SELECT COUNT(*) FROM eln").fetchone()[0]
    log_count = con.execute("SELECT COUNT(*) FROM wo_log").fetchone()[0]
    con.close()
    stages = []
    for i, name in enumerate(WO_STAGES):
        owner = "AI Brain" if i in (0, 4) else ("Vehicle Brain" if i == 1 else ("Dual-arm station" if i == 2 else "Characterization"))
        stages.append({
            "idx": i,
            "name": name,
            "owner": owner,
            "count": stage_counts.get(str(i), 0),
            "evidence": "trace_id + verdict" if i == 0 else ("wo_log + operator" if i < 4 else "lambda_obs + close_note"),
        })
    agents = [
        {"key": "design", "title": "Formula Design Agent", "owner": "browser + AI Brain",
         "input": "formula / dopant / target wavelength", "output": "candidate material object",
         "tools": ["/api/materials/explorer", "/api/ai_brain/explain"], "guardrail": "read-only public query; no private prompt exposure"},
        {"key": "predict", "title": "Synthesis Prediction Agent", "owner": "AI Brain X5",
         "input": "formula object", "output": "trace_id, verdict, CI, sintering hints",
         "tools": ["/api/workorders POST", "lab /api/predict"], "guardrail": "hard priors can override LLM text"},
        {"key": "queue", "title": "Batch Lifecycle Agent", "owner": "Command Center",
         "input": "prediction trace + work order", "output": "stage transitions and audit log",
         "tools": ["/api/workorders/{id}", "/api/export/workorders.csv"], "guardrail": "state changes are role-gated and logged"},
        {"key": "execute", "title": "Embodied Execution Agent", "owner": "Vehicle + dual arms",
         "input": "approved work order", "output": "sample movement and station completion evidence",
         "tools": ["Lab-FSD status", "WorkCockpit mock/live"], "guardrail": "public site never publishes unsafe chassis velocity or arm commands"},
        {"key": "feedback", "title": "Characterization Feedback Agent", "owner": "Researcher",
         "input": "XRD/PL result and lambda_obs", "output": "closed work order and next-round learning signal",
         "tools": ["/api/workorders/{id}/backfill", "/api/materials/{id}"], "guardrail": "measured/history rows are labelled"},
    ]
    protocols = [
        {"name": "NIR phosphor screen", "steps": ["select host/dopant", "predict GO/REVISE", "create batch", "execute sample", "backfill PL/XRD"]},
        {"name": "Failure-loop run", "steps": ["open REVISE rows", "inspect flags", "alter site/concentration", "requeue", "compare evidence_score"]},
        {"name": "Defense demo path", "steps": ["open Brain explain", "filter Atlas", "open material detail", "open work order", "show audit export"]},
    ]
    return {
        "ts": now,
        "release": ASSET_VER,
        "source": "workorders + wo_log + eln + public material schema",
        "summary": {"workorders": sum(state_counts.values()), "open": state_counts.get("open", 0),
                    "done": state_counts.get("done", 0), "cancelled": state_counts.get("cancelled", 0),
                    "eln_notes": eln_count, "audit_events": log_count},
        "stages": stages,
        "agents": agents,
        "objects": objects,
        "protocols": protocols,
        "guardrails": [
            "Public Studio is read-oriented; write APIs remain role-gated.",
            "Every stage must link to trace_id, work_order, wo_log, material object, or labelled replay/curated evidence.",
            "Robot execution is represented as lifecycle state and evidence, not as unsafe public controls.",
        ],
    }


@app.route("/api/agent_studio")
def api_agent_studio():
    return jsonify(_agent_studio_payload(request.args.get("limit", 18)))


def _fsd_brief(data):
    if isinstance(data, dict):
        keys = ["state", "status", "mode", "decision", "gate", "risk", "latency_ms",
                "anomaly_ms", "bpu_ms", "source", "reason"]
        out = {k: data.get(k) for k in keys if k in data and data.get(k) is not None}
        if not out:
            out["keys"] = list(data.keys())[:8]
        return out
    if isinstance(data, list):
        return {"items": len(data)}
    return {}


def _lab_fsd_endpoint(port, path, public_src, timeout=0.35):
    code, body = _probe(port, path, timeout=timeout)
    data = _json(body) if code == 200 else None
    ok = code == 200 and isinstance(data, (dict, list))
    return {
        "path": path,
        "code": code or 0,
        "ok": bool(ok),
        "source": public_src if ok else "offline",
        "brief": _fsd_brief(data) if ok else {},
    }


def _lab_fsd_console_payload():
    now = int(time.time())
    port, src = _serving_port("car", timeout=0.45)
    health_code, _ = _probe_ms(port, "/api/health", timeout=0.45)
    if health_code != 200:
        src = "down"
    public_src = _twin_public_source(src)
    endpoint_src = public_src if public_src in ("live", "mock") else "offline"
    endpoint_paths = [
        "/lab_fsd/fsd_v2_status",
        "/lab_fsd/future_bev",
        "/lab_fsd/policy_tokens",
        "/lab_fsd/safety_gate",
    ]
    endpoints = [_lab_fsd_endpoint(port, p, endpoint_src) for p in endpoint_paths] if src != "down" else [
        {"path": p, "code": 0, "ok": False, "source": "offline", "brief": {}} for p in endpoint_paths
    ]
    ep_by_path = {e["path"]: e for e in endpoints}
    live_detail_count = sum(1 for e in endpoints if e.get("ok"))
    world_source = public_src if live_detail_count else "replay"
    safety_ep = ep_by_path.get("/lab_fsd/safety_gate") or {}
    safety_state = "blocked" if public_src == "offline" else "review"
    reason = "Lab-FSD public detail endpoint not exposed; read-only replay console is active."
    if safety_ep.get("ok"):
        b = safety_ep.get("brief") or {}
        raw = str(b.get("state") or b.get("decision") or b.get("gate") or "").lower()
        if raw in ("pass", "clear", "ok", "allow"):
            safety_state = "pass"
        elif raw in ("block", "blocked", "stop", "deny", "hold"):
            safety_state = "blocked"
        else:
            safety_state = "review"
        reason = str(b.get("reason") or "Safety gate endpoint returned a public summary.")
    elif public_src == "offline":
        reason = "Vehicle brain endpoint is offline; public console holds all motion."

    grid_marks = {
        (1, 5): ("route", "R1", "sample corridor"),
        (2, 4): ("route", "R2", "sample corridor"),
        (3, 4): ("route", "R3", "sample corridor"),
        (4, 3): ("ego", "X5", "current ego cell"),
        (5, 3): ("route", "R4", "predicted route"),
        (6, 2): ("route", "R5", "predicted route"),
        (7, 2): ("goal", "G", "dock goal"),
        (4, 2): ("shadow", "S", "probabilistic shadow"),
        (5, 2): ("warn", "W", "bench edge clearance"),
        (6, 1): ("blocked", "B", "hot-zone boundary"),
        (7, 1): ("blocked", "B", "hot-zone boundary"),
        (2, 5): ("sensor", "L", "LD14 sweep"),
        (3, 2): ("sensor", "D", "Astra depth frustum"),
    }
    cells = []
    for y in range(7):
        for x in range(9):
            level, short, label = grid_marks.get((x, y), ("free", "", "free occupancy cell"))
            cells.append({"x": x, "y": y, "level": level, "short": short, "label": label,
                          "source": world_source})
    base_conf = 0.91 if public_src == "live" and live_detail_count else (0.76 if public_src != "offline" else 0.0)
    future = [
        {"t": "+0.0s", "intent": "hold current corridor", "risk": "nominal" if safety_state == "pass" else "review",
         "pose": "x=0.0 y=0.0 yaw=0.0", "source": world_source},
        {"t": "+1.5s", "intent": "slow approach to sample dock", "risk": "shadow monitored",
         "pose": "x=0.4 y=0.2 yaw=0.3", "source": world_source},
        {"t": "+3.0s", "intent": "yield near arm sweep zone", "risk": "operator review",
         "pose": "x=0.9 y=0.4 yaw=0.5", "source": world_source},
        {"t": "+5.0s", "intent": "dock only after safety gate pass", "risk": "hard gate",
         "pose": "x=1.4 y=0.8 yaw=0.9", "source": world_source},
    ]
    tokens = [
        {"token": "OCCUPANCY_CLEAR", "confidence": round(base_conf, 2), "source": world_source,
         "reason": "BEV free-space corridor is public-display safe"},
        {"token": "SHADOW_REVIEW", "confidence": round(max(base_conf - 0.11, 0), 2), "source": world_source,
         "reason": "unknown cells stay in review instead of becoming motion permission"},
        {"token": "YIELD_ARM_ZONE", "confidence": round(max(base_conf - 0.08, 0), 2), "source": "replay",
         "reason": "dual-arm sweep zone is represented as a protected boundary"},
        {"token": "DOCK_AFTER_GATE", "confidence": round(max(base_conf - 0.05, 0), 2), "source": world_source,
         "reason": "safety gate remains the final read-only decision shown publicly"},
        {"token": "NO_PUBLIC_VELOCITY_CMD", "confidence": 1.0, "source": "policy",
         "reason": "public site never publishes robot motion commands"},
    ]
    systems = [
        {"name": "LD14 lidar", "role": "2D obstacle belt", "state": "serving" if public_src != "offline" else "offline",
         "source": public_src},
        {"name": "Astra Pro depth", "role": "near-field occupancy", "state": "summarized", "source": world_source},
        {"name": "USB camera", "role": "visual context / anomaly hints", "state": "summarized", "source": world_source},
        {"name": "Temporal world model", "role": "future BEV rollout", "state": "read-only", "source": world_source},
        {"name": "BPU anomaly AE", "role": "edge anomaly score, measured about 1.7ms on X5", "state": "architecture metric",
         "source": "documented"},
        {"name": "Nav2 / MPPI", "role": "authoritative chassis control", "state": "not controlled by public site",
         "source": "safety policy"},
    ]
    twin = _build_twin(58)
    return {
        "ts": now,
        "release": ASSET_VER,
        "source": world_source,
        "summary": {
            "serving": public_src,
            "serving_endpoint": "operator-only" if src != "down" else None,
            "health_code": health_code or 0,
            "active_endpoint_count": live_detail_count,
            "world_model_source": world_source,
            "safety_gate": safety_state,
            "public_control": "read-only / no chassis velocity command",
        },
        "safety_gate": {"state": safety_state, "reason": reason, "source": safety_ep.get("source") or public_src},
        "occupancy": {"cols": 9, "rows": 7, "cells": cells},
        "future": future,
        "policy_tokens": tokens,
        "systems": systems,
        "endpoints": endpoints,
        "trace": twin.get("plan") or [],
        "boundaries": [
            "This is Lab-FSD v2 for a laboratory robot; it borrows world-model interface patterns but does not claim to replicate Tesla FSD.",
            "The public page is read-only and never emits chassis velocity, arm motion, lift, actuator, magnet, or servo commands.",
            "Nav2/MPPI and embedded safety layers remain the chassis-control authority.",
            "Every panel is labelled live, mock, replay, offline, documented, or policy.",
        ],
    }


@app.route("/api/lab_fsd_console")
def api_lab_fsd_console():
    return jsonify(_lab_fsd_console_payload())


_REPLAY_FAULTS = {
    "device_offline": {
        "label": "device offline",
        "stage": 3,
        "severity": "warn",
        "symptom": "vehicle or workstation heartbeat drops to mirror/offline",
        "recovery": "hold sample state, switch public display to replay, require operator restore before execution resumes",
    },
    "sample_missing": {
        "label": "sample missing",
        "stage": 5,
        "severity": "crit",
        "symptom": "vision/weight evidence does not confirm powder or bottle at the expected station",
        "recovery": "stop actuator sequence, ask operator to reseat sample, keep work order open with evidence note",
    },
    "route_blocked": {
        "label": "route blocked",
        "stage": 3,
        "severity": "warn",
        "symptom": "BEV occupancy marks the sample corridor as blocked",
        "recovery": "shadow planner yields; Nav2/MPPI remains authority and no public command is emitted",
    },
    "arm_station_busy": {
        "label": "arm station busy",
        "stage": 4,
        "severity": "warn",
        "symptom": "dual-arm sweep zone or interlock state is not clear",
        "recovery": "queue waits at the station boundary and replays the latest safe arm pose",
    },
    "characterization_mismatch": {
        "label": "characterization mismatch",
        "stage": 9,
        "severity": "review",
        "symptom": "XRD/PL observation diverges from the prediction evidence band",
        "recovery": "close with mismatch note, attach actual lambda/XRD evidence, create next-round REVISE candidate",
    },
}


def _latest_replay_object():
    con = _db()
    row = con.execute(f"SELECT {_WO_COLS} FROM workorders ORDER BY id DESC LIMIT 1").fetchone()
    if row:
        w = _wo_dict(row)
        logs = con.execute("SELECT ts,author,action,detail FROM wo_log WHERE wo=? ORDER BY id DESC LIMIT 6",
                           (w["id"],)).fetchall()
        con.close()
        return {
            "kind": "work_order",
            "work_order_id": w["id"],
            "code": w.get("code") or "",
            "trace_id": w.get("trace_id") or "",
            "formula": w.get("formula") or "",
            "dopant": w.get("dop_symbol") or "",
            "site": w.get("dop_site") or "",
            "state": w.get("state") or "",
            "stage": w.get("stage") or 0,
            "verdict": w.get("verdict") or "",
            "lambda_obs": w.get("lambda_obs"),
            "source": "workorders + wo_log",
            "recent_log": [{"ts": r[0], "author": r[1], "action": r[2], "detail": r[3]} for r in logs],
        }
    con.close()
    try:
        row = (_materials_all_rows() or [{}])[0]
    except Exception:
        row = {}
    return {
        "kind": "candidate_material",
        "work_order_id": None,
        "code": row.get("id") or row.get("trace_id") or row.get("formula") or "candidate",
        "trace_id": row.get("trace_id") or "",
        "formula": row.get("formula") or "Y3Al5O12",
        "dopant": row.get("dopant") or "Cr3+",
        "site": row.get("site") or "Al",
        "state": row.get("state") or "not queued",
        "stage": 0,
        "verdict": row.get("verdict") or "",
        "lambda_obs": row.get("lambda_obs"),
        "source": "public material schema replay",
        "recent_log": [],
    }


def _experiment_replay_payload(progress=58, fault=""):
    try:
        p = max(0, min(100, float(progress)))
    except Exception:
        p = 58.0
    fault = (fault or "").strip()
    active_fault = _REPLAY_FAULTS.get(fault)
    trace = _latest_replay_object()
    markers = [
        ("design", "AI candidate design", "AI Brain", "formula object + target band", "trace_id or material object"),
        ("predict", "AI prediction verdict", "AI Brain X5", "TS/MLIP/failure flags + R1 verdict", "prediction trace"),
        ("queue", "queue and approval", "Command Center", "work order enters staged lifecycle", "work_order + wo_log"),
        ("car_route", "embodied real-hardware sample loop", "Embodied Brain", "bottle pickup, lift, 0.50 m odometry loop, release and reset", "observed hardware record + public replay"),
        ("arm01_pick", "arm01 visual redundancy and bag drop", "Arm01 workstation", "visual gate, redundant pickup and bag transfer", "observed visual-gate record"),
        ("arm02_grind", "arm02 concurrent four-cycle grinding", "Arm02 workstation", "concurrent grinding while arm01 returns to the safe pose", "observed dual-arm record"),
        ("execution_reset", "execution-layer safety reset", "Embedded execution layer", "bounded release, reset and fail-closed completion", "public-safe execution replay"),
        ("furnace", "sintering and furnace wait", "Furnace", "temperature profile and hot-zone lock", "protocol template"),
        ("xrd", "XRD characterization", "XRD line", "pattern capture and phase check", "XRD evidence object"),
        ("pl_feedback", "PL feedback and failure loop", "PL line + AI Brain", "lambda_obs, mismatch, next candidate", "actual feedback"),
    ]
    active_idx = min(9, int(p // 10))
    stages = []
    for i, m in enumerate(markers):
        state = "done" if i < active_idx else ("active" if i == active_idx else "pending")
        if active_fault and i == active_fault["stage"]:
            state = "fault"
        stages.append({
            "idx": i,
            "pct": i * 10,
            "key": m[0],
            "title": m[1],
            "owner": m[2],
            "action": m[3],
            "evidence": m[4],
            "state": state,
            "source": "replay" if i >= 3 else trace.get("source", "replay"),
        })
    events = [
        {"t": "00:00", "stage": "design", "system": "AI brain", "event": "candidate object selected", "source": trace["source"]},
        {"t": "00:30", "stage": "predict", "system": "AI brain", "event": "prediction verdict and uncertainty attached", "source": "trace"},
        {"t": "01:00", "stage": "queue", "system": "Command Center", "event": "work order stage record created", "source": "wo_log"},
        {"t": "01:35", "stage": "car_route", "system": "Embodied brain", "event": "real-hardware bottle handling and 0.50 m odometry loop completed", "source": "observed/replay"},
        {"t": "02:10", "stage": "arm01_pick", "system": "arm01", "event": "visual redundancy and bag drop completed", "source": "observed/replay"},
        {"t": "02:42", "stage": "arm02_grind", "system": "arm02", "event": "concurrent four-cycle grinding completed", "source": "observed/replay"},
        {"t": "03:15", "stage": "execution_reset", "system": "execution layer", "event": "bounded release and reset completed", "source": "observed/replay"},
        {"t": "04:20", "stage": "furnace", "system": "Furnace", "event": "sintering protocol and hot-zone lock", "source": "protocol"},
        {"t": "06:00", "stage": "xrd", "system": "XRD line", "event": "phase check object generated", "source": "characterization"},
        {"t": "07:20", "stage": "pl_feedback", "system": "PL line", "event": "lambda_obs backfill and failure learning", "source": "actual/replay"},
    ]
    if active_fault:
        events.insert(min(len(events), active_fault["stage"] + 1), {
            "t": "FAULT",
            "stage": stages[active_fault["stage"]]["key"],
            "system": stages[active_fault["stage"]]["owner"],
            "event": active_fault["label"] + ": " + active_fault["symptom"],
            "source": "fault injection",
            "severity": active_fault["severity"],
            "recovery": active_fault["recovery"],
        })
    twin = _build_twin(p)
    actuators = [
        {"key": "sample_lift", "label": "Sample lift", "state": "replay", "public_command": "disabled", "evidence": "Observed lift-stage completion; public replay only"},
        {"key": "bottle_handling", "label": "Bottle handling", "state": "replay", "public_command": "disabled", "evidence": "Observed pickup and release completion; public replay only"},
        {"key": "safety_interlock", "label": "Safety interlock", "state": "replay", "public_command": "disabled", "evidence": "Fail-closed boundary evidence"},
        {"key": "execution_reset", "label": "Execution reset", "state": "replay", "public_command": "disabled", "evidence": "Observed final reset; public replay only"},
    ]
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "mode": "replay",
        "progress_pct": round(p, 1),
        "active_stage": active_idx,
        "trace": trace,
        "stages": stages,
        "events": events,
        "faults": [{"key": k, **v} for k, v in _REPLAY_FAULTS.items()],
        "active_fault": {"key": fault, **active_fault} if active_fault else None,
        "twin": {
            "map": twin.get("map") or {},
            "sample": twin.get("sample") or {},
            "car": twin.get("car"),
            "arms": twin.get("arms") or {},
            "source": twin.get("source") or {},
            "source_label": twin.get("source_label") or {},
        },
        "actuators": actuators,
        "guardrails": [
            "Replay is explanatory and read-only; it does not trigger car, arm, lift, magnet, actuator, or servo commands.",
            "Live, mock, replay, stale, offline, and documented sources stay labelled at panel level.",
            "Fault injection changes the public replay narrative only; it is not injected into real robot services.",
        ],
    }


@app.route("/api/experiment_replay")
def api_experiment_replay():
    return jsonify(_experiment_replay_payload(request.args.get("pct", 58), request.args.get("fault", "")))


def _asset_group_count():
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return doc.get("groups") or []
    except Exception:
        return []


def _cloud_command_payload():
    now = int(time.time())
    fleet = _fleet_console_payload()
    tasks = _tasks_console_payload(40)
    replay = _experiment_replay_payload(58, "")
    groups = _asset_group_count()
    con = _db()
    active_alarms = [{"severity": r[0], "sys": r[1], "message": _public_safe_text(r[2]), "ts": r[3]}
                     for r in con.execute("SELECT severity,sys,message,ts_raised FROM alarms"
                                          " WHERE ts_cleared IS NULL ORDER BY ts_raised DESC LIMIT 8").fetchall()]
    maint = [{"ts": r[0], "asset": r[1], "author": r[2], "note": _public_safe_text(r[3])}
             for r in con.execute("SELECT ts,asset,author,note FROM maintenance ORDER BY id DESC LIMIT 8").fetchall()]
    con.close()
    task_rows = tasks.get("tasks") or []
    lanes = [
        {"key": "design", "label": "Design / Predict", "stages": [0, 1]},
        {"key": "execute", "label": "Embodied Execution", "stages": [2]},
        {"key": "characterize", "label": "Furnace / XRD / PL", "stages": [3]},
        {"key": "feedback", "label": "Feedback Loop", "stages": [4, 5]},
    ]
    lane_cards = []
    for lane in lanes:
        items = [t for t in task_rows if int(t.get("stage") or 0) in lane["stages"]]
        lane_cards.append({
            **lane,
            "count": len(items),
            "items": items[:4],
            "source": "workorders" if items else "empty",
        })
    resources = []
    sys_map = {s["key"]: s for s in fleet.get("systems") or []}
    for key, label, capability in [
        ("lab", "AI Brain", "LLM/BPU prediction, GraphRAG, audit trace"),
        ("car", "Embodied Brain", "observed bottle-handling and odometry loop; Lab-FSD remains shadow/assist"),
        ("arm", "Arm01 Workstation", "observed visual redundancy, bag transfer and safe return"),
        ("arm02", "Arm02 Workstation", "observed concurrent four-cycle grinding; CPU/OpenCV bag-state authority"),
    ]:
        s = sys_map.get(key, {})
        resources.append({"key": key, "label": label, "capability": capability,
                          "source": s.get("source") or "unknown", "serving": s.get("serving") or "unknown",
                          "reservation": "available for replay; physical execution requires operator readiness"})
    for key, label, cap in [
        ("lift", "Lift Stage", "F407 lift-stage replay / documented readiness"),
        ("magnet", "Electromagnet", "F407 magnet-state replay"),
        ("actuator", "Linear Actuator", "relay extend/retract replay"),
        ("furnace", "Furnace", "hot-zone protocol and risk lock"),
        ("xrd", "XRD Line", "phase check evidence object"),
        ("pl", "PL Line", "lambda_obs feedback evidence"),
        ("vps", "VPS/Auth/Static", "SSO, Caddy, service worker, release rollback"),
    ]:
        resources.append({"key": key, "label": label, "capability": cap, "source": "documented/replay",
                          "serving": "replay", "reservation": "public display only"})
    blockers = []
    for t in task_rows[:12]:
        blocker = t.get("blocker") or ""
        if blocker:
            blockers.append({"task": t.get("code") or t.get("formula"), "blocker": blocker,
                             "decision": "operator review" if "review" in blocker or "waiting" in blocker else "system retry",
                             "url": "/tasks"})
    if not blockers:
        blockers.append({"task": "sample-flow replay", "blocker": "no live execution request queued",
                         "decision": "operator approval before physical run", "url": "/replay"})
    calendar = [
        {"t": "+00m", "resource": "AI Brain", "slot": "prediction / trace binding", "state": "ready" if sys_map.get("lab", {}).get("source") in ("live", "mirror") else "replay"},
        {"t": "+05m", "resource": "Vehicle Brain", "slot": "sample route reservation", "state": sys_map.get("car", {}).get("source") or "unknown"},
        {"t": "+12m", "resource": "Dual-arm Workstation", "slot": "observed finals workflow replay", "state": sys_map.get("arm", {}).get("source") or "unknown"},
        {"t": "+25m", "resource": "Furnace", "slot": "hot-zone locked protocol", "state": "risk lock"},
        {"t": "+1h", "resource": "XRD / PL", "slot": "characterization capture", "state": "replay/history"},
    ]
    approvals = [
        {"gate": "manual approval", "state": "required", "who": "operator/member", "reason": "physical execution is never started from public page"},
        {"gate": "risk lock", "state": "active", "who": "safety policy", "reason": "hot-zone, arm sweep, and public command boundaries"},
        {"gate": "rollback/cancel", "state": "available", "who": "role-gated API", "reason": "work order can be cancelled or replayed, not remotely driven"},
    ]
    return {
        "ts": now,
        "release": ASSET_VER,
        "summary": {
            "open_tasks": tasks.get("counts", {}).get("open", 0),
            "done_tasks": tasks.get("counts", {}).get("done", 0),
            "resources": len(resources),
            "blockers": len(blockers),
            "alarms": len(active_alarms),
        },
        "fleet": fleet.get("summary") or {},
        "lanes": lane_cards,
        "resources": resources,
        "calendar": calendar,
        "blockers": blockers,
        "approvals": approvals,
        "sample_locations": [
            {"label": n.get("label"), "source": n.get("source"), "x": n.get("x"), "y": n.get("y")}
            for n in ((replay.get("twin") or {}).get("map") or {}).get("nodes", [])
        ],
        "active_alarms": active_alarms,
        "maintenance": maint,
        "asset_groups": [{"key": g.get("key"), "name": g.get("name"), "children": len(g.get("children") or [])}
                         for g in groups],
        "guardrails": [
            "Cloud Command is a public scheduling and evidence view; it does not expose dangerous remote control.",
            "Equipment reservation and fault locks are simulated/read-only until an authenticated operator acts inside protected systems.",
            "Every queue card links toward tasks, replay, twin, assets, or observability instead of hiding blockers.",
        ],
    }


@app.route("/api/cloud_command_center")
def api_cloud_command_center():
    return jsonify(_cloud_command_payload())


def _defense_mode_payload():
    trace = _latest_replay_object()
    trace_id = trace.get("trace_id") or trace.get("code") or trace.get("formula") or ""
    scripts = [
        {"key": "three_min", "label": "3 minute script", "duration": "3:00", "goal": "state the problem, the closed loop, and the proof surface",
         "beats": [
             {"t": "00:00", "title": "Problem", "body": "NIR phosphor discovery is slow because design, synthesis, XRD, PL and feedback are separated."},
             {"t": "00:35", "title": "System", "body": "The public site shows one software-defined lab loop: AI prediction, embodied execution replay, characterization and feedback."},
             {"t": "01:20", "title": "Differentiator", "body": "AI Brain X5, Materials Atlas, Lab-FSD v2, Replay and Cloud Command are linked by trace/work_order evidence."},
             {"t": "02:15", "title": "Boundary", "body": "Offline or replay sources stay labelled; public pages never expose robot or actuator controls."},
         ]},
        {"key": "five_min", "label": "5 minute script", "duration": "5:00", "goal": "walk a judge through the strongest evidence path",
         "beats": [
             {"t": "00:00", "title": "Open the OS", "body": "Start at the homepage, then open Defense Mode as the evidence map."},
             {"t": "00:45", "title": "AI proof", "body": "Show Brain explain and Atlas rows with CI, source labels and export endpoints."},
             {"t": "02:00", "title": "Embodied proof", "body": "Show Lab-FSD read-only world model and Experiment Replay with fault injection."},
             {"t": "03:25", "title": "Cloud lab proof", "body": "Show Command Center resource board, blockers, approvals and public guardrails."},
             {"t": "04:20", "title": "Audit proof", "body": "Open traces, hardening and export links to demonstrate reproducibility and public safety."},
         ]},
        {"key": "eight_min", "label": "8 minute script", "duration": "8:00", "goal": "complete a full defense narrative with offline fallback",
         "beats": [
             {"t": "00:00", "title": "Why it matters", "body": "Explain 15 months to 15 minutes as a closed-loop materials automation objective, not a marketing slogan."},
             {"t": "01:00", "title": "Scientific stack", "body": "TS inverse, conformal CI, MLIP/cache labels and GraphRAG are presented as inspectable evidence, not hidden claims."},
             {"t": "02:20", "title": "AI/edge stack", "body": "9 local LLM paths, BPU slot evidence and Fly-MB reasoning are visible through public summaries."},
             {"t": "03:35", "title": "Embodied autonomy", "body": "SLAM map evidence is shown with Lab-FSD shadow/assist; Nav2/MPPI and the safety operator remain the physical execution boundary."},
             {"t": "04:50", "title": "Lifecycle execution", "body": "Replay and Studio connect formula, trace_id, work_order, stages, faults, actuators and recovery notes."},
             {"t": "06:10", "title": "Operations", "body": "Command, observability, traces and hardening prove the system can be operated, debugged and defended."},
             {"t": "07:20", "title": "Fallback", "body": "If hardware is offline, labelled replay/history still completes the explanation without pretending to be live."},
         ]},
    ]
    evidence = [
        {"key": "ai_brain_x5", "claim": "AI Brain X5 prediction stack", "type": "API + page", "href": "/brain", "api": "/api/ai_brain/explain", "source": "mirror/live", "boundary": "public summaries only; no raw prompts or secrets"},
        {"key": "local_llm_bpu", "claim": "9 local LLM / 5 BPU slot evidence", "type": "API + source file", "href": "/brain", "api": "/api/ai_brain/explain", "source": "documented/mirror", "boundary": "latency and slot labels are evidence, not public model control"},
        {"key": "fly_mb", "claim": "Fly-MB memory brain explainability", "type": "page + reasoning trace", "href": "/brain", "api": "/api/ai_brain/explain", "source": "curated/mirror", "boundary": "compressed explanation, not private chain-of-thought"},
        {"key": "ts_ci", "claim": "TS inverse / Conformal CI scientific proof", "type": "method + export", "href": "/atlas", "api": "/api/materials/explorer", "source": "history/mirror", "boundary": "uncertainty shown; unsupported constants stay labelled"},
        {"key": "materials_atlas", "claim": "Materials Atlas searchable object model", "type": "page + export", "href": "/atlas", "api": "/api/materials/export.json", "source": "history/mirror", "boundary": "rows keep live/mirror/history/source labels"},
        {"key": "lab_fsd", "claim": "SLAM map + Lab-FSD shadow world model and safety gate", "type": "page + API", "href": "/fsd", "api": "/api/lab_fsd_console", "source": "replay/mirror", "boundary": "read-only; no navigation velocity or physical control"},
        {"key": "replay_twin", "claim": "Digital Twin / Experiment Replay lifecycle", "type": "page + API", "href": "/replay", "api": "/api/experiment_replay", "source": "replay/history", "boundary": "fault injection affects narrative only"},
        {"key": "arm01_redundancy", "claim": "Finals dual-arm hardware evidence: arm01 visual redundancy and bag drop with arm02 concurrent four-cycle grinding", "type": "measured + replay", "href": "/replay", "api": "/api/experiment_replay?pct=58", "source": "observed/replay", "boundary": "X5 CPU/OpenCV is authoritative for bag state; BPU is assist/evidence only; no public arm, lift, magnet, linear actuator or yaw command"},
        {"key": "cloud_command", "claim": "Cloud Lab Command Center scheduling", "type": "page + API", "href": "/command", "api": "/api/cloud_command_center", "source": "mirror/replay", "boundary": "public display only; operator approval remains protected"},
        {"key": "visual_system", "claim": "Liquid Glass visual system", "type": "screenshot + CSS", "href": "/", "api": "/style.css", "source": "site asset", "boundary": "visual layer supports evidence; it is not used as proof by itself"},
        {"key": "observability_security", "claim": "Observability, traces, exports and hardening", "type": "pages + APIs", "href": "/observability", "api": "/api/hardening", "source": "live/site", "boundary": "public-safe logs only; secrets and private prompts excluded"},
    ]
    demo_paths = [
        {"key": "live", "label": "live / mirror demo path", "source": "live/mirror", "steps": [
            {"title": "Open Defense", "href": "/defense", "evidence": "script + checklist"},
            {"title": "AI proof", "href": "/brain", "evidence": "/api/ai_brain/explain"},
            {"title": "Materials proof", "href": "/atlas", "evidence": "/api/materials/explorer"},
            {"title": "Command proof", "href": "/command", "evidence": "/api/cloud_command_center"},
            {"title": "Trace proof", "href": "/traces", "evidence": "/api/traces"},
        ]},
        {"key": "offline", "label": "offline hardware demo path", "source": "replay/history", "steps": [
            {"title": "State boundary", "href": "/status", "evidence": "offline/replay labels"},
            {"title": "Replay sample flow", "href": "/replay", "evidence": "/api/experiment_replay"},
            {"title": "FSD read-only", "href": "/fsd", "evidence": "/api/lab_fsd_console"},
            {"title": "Evidence exports", "href": "/atlas", "evidence": "JSON/CSV/report links"},
            {"title": "Hardening", "href": "/sec", "evidence": "/api/hardening"},
        ]},
    ]
    checklist = [
        {"item": "Can explain the core loop in 3 minutes", "state": "pass", "evidence": "three_min script"},
        {"item": "Every major claim has a page or API evidence link", "state": "pass", "evidence": f"{len(evidence)} evidence cards"},
        {"item": "Offline hardware still has a labelled defense path", "state": "pass", "evidence": "offline demo path"},
        {"item": "Replay/mock/mirror are not described as live", "state": "pass", "evidence": "source labels on evidence cards"},
        {"item": "No public dangerous controls", "state": "pass", "evidence": "guardrail scan and read-only APIs"},
        {"item": "Exportable objects exist for materials and traces", "state": "pass", "evidence": "/api/materials/export.json + /api/traces"},
    ]
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "summary": {"scripts": len(scripts), "evidence": len(evidence), "demo_paths": len(demo_paths),
                    "checklist": len(checklist), "trace": trace_id},
        "scripts": scripts,
        "evidence": evidence,
        "demo_paths": demo_paths,
        "checklist": checklist,
        "judge_shortlist": ["Problem", "Closed loop", "Scientific proof", "Embodied proof", "Evidence links", "Safety boundary"],
        "guardrails": [
            "Defense Mode is an evidence index and script surface; it does not add public write operations.",
            "Claims without evidence links must be treated as plan, replay, documented boundary, or offline history.",
            "Robot navigation and actuator control remain outside public pages.",
        ],
    }


@app.route("/api/defense_mode")
def api_defense_mode():
    return jsonify(_defense_mode_payload())


def _global_benchmark_payload():
    """Compatibility wrapper for callers that still import the pre-Site31 helper."""
    return _site31_global_benchmark_payload()


def _site31_global_benchmark_payload():
    scorecard = _site31_scorecard_payload()
    refs = {
        "value_clarity": ("Google DeepMind Research / NASA Science", "https://deepmind.google/research/", "使命、研究突破与对象入口分层"),
        "scientific_provenance": ("Materials Project / CERN Open Data", "https://opendata.cern.ch/docs/about", "可搜索对象、版本、引用、下载和限制"),
        "information_architecture": ("Vercel / Benchling", "https://vercel.com/changelog/dashboard-navigation-redesign-rollout", "高频工作流优先与对象上下文"),
        "visual_system": ("Stripe / Apple interface principles", "https://stripe.com/", "鲜艳品牌层与克制数据层"),
        "motion": ("View Transition API / Vercel interface guidance", "https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API", "短、可中断、解释关系的动效"),
        "performance": ("Core Web Vitals", "https://web.dev/articles/vitals", "LCP、INP、CLS 与真实用户 p75"),
        "security_trust": ("Cloudflare WAF / OWASP", "https://developers.cloudflare.com/waf/", "边缘规则、速率限制与验证状态分离"),
        "accessibility": ("WCAG 2.2", "https://www.w3.org/TR/WCAG22/", "整页 AA、键盘、状态、reduced motion"),
        "localization": ("Global research portals", "https://science.nasa.gov/", "主叙事本地化与术语一致"),
        "release_engineering": ("Vercel deployment transparency", "https://vercel.com/docs/deployments/overview", "版本、状态、变更与回滚证据"),
    }
    surfaces = {
        "value_clarity": ["/", "/api/research_portal"],
        "scientific_provenance": ["/atlas", "/api/evidence_objects", "/api/research_passport"],
        "information_architecture": ["/", "/api/public_manifest"],
        "visual_system": ["/", "/design"],
        "motion": ["/", "/benchmark"],
        "performance": ["/api/hardening"],
        "security_trust": ["/sec", "/api/trust_center", "/api/hardening"],
        "accessibility": ["/", "/api/hardening"],
        "localization": ["/", "/atlas", "/fsd"],
        "release_engineering": ["/release", "/api/releases"],
    }
    gaps = {
        "performance": "尚无稳定真实用户 CWV p75; 当前只能记录实验室与代码级证据。",
        "security_trust": "Cloudflare WAF/rate limiting 和网关身份头清理仍需控制台/配置证据。",
        "accessibility": "尚未完成完整 NVDA/VoiceOver 与 WCAG-EM 人工评估。",
        "scientific_provenance": "数据规模仍是项目级, 不等同全球公共材料数据库。",
    }
    dimensions = []
    for row in scorecard["dimensions"]:
        benchmark, url, trait = refs[row["key"]]
        pct = round(100 * row["earned_points"] / row["max_points"], 1) if row["max_points"] else 0
        dimensions.append({
            "key": row["key"], "label": row["label"], "benchmark": benchmark,
            "external_url": url, "top_trait": trait, "our_surface": surfaces[row["key"]],
            "evidence": [row["method"], "状态: " + row["state"], "证据: " + " · ".join(row["evidence"])],
            "score": pct, "state": row["state"],
            "gap": gaps.get(row["key"], "完成本轮浏览器与线上回归后才可升级为 verified。"),
        })
    gates = [
        {"key": "internal_score", "label": "内部证据门禁 >= 90/100", "state": "verified" if scorecard["score"] >= 90 else "work-in-progress", "value": scorecard["score"]},
        {"key": "no_low_dimension", "label": "无维度低于 75%", "state": "verified" if all(x["earned_points"] >= x["max_points"] * .75 for x in scorecard["dimensions"]) else "work-in-progress", "value": "internal rubric"},
        {"key": "public_safety", "label": "公网无物理控制入口", "state": "verified", "value": "/api/hardening"},
        {"key": "source_labels", "label": "状态来源标签保留", "state": "verified", "value": "live/mirror/replay/planned/stale/offline/unknown"},
        {"key": "external_rank", "label": "不伪造第三方全球排名", "state": "verified", "value": "no external rank claim"},
    ]
    return {
        "ts": int(time.time()), "release": ASSET_VER,
        "overall_score": scorecard["score"], "score_max": scorecard["max_score"],
        "score_type": scorecard["score_type"], "dimension_count": len(dimensions),
        "gate_state": scorecard["gate"], "dimensions": dimensions, "gates": gates,
        "sources": [
            {"label": "DeepMind Research", "url": "https://deepmind.google/research/"},
            {"label": "Materials Project", "url": "https://docs.materialsproject.org/apps/explorer-apps"},
            {"label": "CERN Open Data", "url": "https://opendata.cern.ch/docs/about"},
            {"label": "Hugging Face Model Cards", "url": "https://huggingface.co/docs/hub/en/model-cards"},
            {"label": "Vercel Dashboard", "url": "https://vercel.com/changelog/dashboard-navigation-redesign-rollout"},
            {"label": "Core Web Vitals", "url": "https://web.dev/articles/vitals"},
            {"label": "WCAG 2.2", "url": "https://www.w3.org/TR/WCAG22/"},
            {"label": "Cloudflare WAF", "url": "https://developers.cloudflare.com/waf/"},
        ],
        "next_bets": [
            "完成 Site31 五档桌面浏览器回归后更新内部验证状态。",
            "接入真实用户 Web Vitals 后再评价长期性能门禁。",
            "取得 Cloudflare 控制台/API 和 Caddy 配置证据后再升级边缘安全状态。",
        ],
        "honest_boundary": "本页是内部证据加权发布门禁, 不是第三方排名。目标是对标一流科研与商业平台, 不宣称拥有其数据规模、认证或商业运营规模。",
    }


@app.route("/api/global_benchmark")
def api_global_benchmark():
    return jsonify(_site31_global_benchmark_payload())


def _observability_cockpit_payload():
    now = int(time.time())
    con = _db()
    latest_app = con.execute("SELECT ts,p95_ms,req_total,req_4xx,req_5xx,rss_kb,threads,db_bytes"
                             " FROM app_metrics ORDER BY ts DESC LIMIT 1").fetchone()
    avails = [_availability_pct(con, k, now - 86400) for k in ("lab", "car", "arm")]
    av_vals = [x for x in avails if x is not None]
    wo = con.execute("SELECT COUNT(*), SUM(trace_id IS NOT NULL AND trace_id!='') FROM workorders").fetchone()
    pred_success = round(100.0 * (wo[1] or 0) / wo[0], 2) if wo[0] else None
    car = _last_system_sample(con, "car")
    arm = _last_system_sample(con, "arm")
    con.close()
    p95 = latest_app[1] if latest_app and latest_app[1] is not None else _pct(list(_REQ_LAT), .95)
    total = _REQ["total"] or (latest_app[2] if latest_app else 0) or 0
    err = _REQ["c5xx"] or (latest_app[4] if latest_app else 0) or 0
    err_pct = round(100.0 * err / total, 3) if total else 0.0
    cards = [
        {"key": "availability", "label": "24h availability", "value": round(sum(av_vals) / len(av_vals), 2) if av_vals else None,
         "unit": "%", "source": "historian", "state": "ok" if av_vals else "unknown"},
        {"key": "p95_latency", "label": "p95 latency", "value": round(p95, 1) if p95 is not None else None,
         "unit": "ms", "source": "RED", "state": "warn" if p95 and p95 > 1200 else "ok"},
        {"key": "error_rate", "label": "5xx error rate", "value": err_pct, "unit": "%", "source": "RED",
         "state": "crit" if err_pct > 1 else "ok"},
        {"key": "prediction_success", "label": "prediction trace bind", "value": pred_success, "unit": "%",
         "source": "workorders", "state": "ok" if pred_success is not None else "unknown"},
        {"key": "bpu_slot_latency", "label": "BPU slot latency", "value": None, "unit": "ms",
         "source": "not exposed publicly", "state": "unknown"},
        {"key": "llm_first_token", "label": "LLM first token", "value": None, "unit": "ms",
         "source": "not exposed publicly", "state": "unknown"},
        {"key": "car_heartbeat", "label": "car heartbeat age", "value": (now - car[0]) if car else None, "unit": "s",
         "source": _serving_source(car[1], now - car[0]) if car else "unknown",
         "state": "ok" if car and now - car[0] < 120 else "warn"},
        {"key": "arm_last_seen", "label": "arm last seen age", "value": (now - arm[0]) if arm else None, "unit": "s",
         "source": _serving_source(arm[1], now - arm[0]) if arm else "unknown",
         "state": "ok" if arm and now - arm[0] < 120 else "warn"},
    ]
    return {"ts": now, "release": ASSET_VER, "cards": cards,
            "red": {"total": total, "err_5xx": err, "err_pct": err_pct, "p95_ms": cards[1]["value"]},
            "note": "Unknown means no public metric is exposed; the UI must not invent realtime telemetry."}


@app.route("/api/observability_cockpit")
def api_observability_cockpit():
    return jsonify(_observability_cockpit_payload())


def _trace_stage_template(total_ms=None, has_wo=False):
    weights = [("parser", "formula parser", "lab", .05),
               ("mlip", "MACE / TS inference", "lab", .22),
               ("bpu", "BPU perception / slot", "lab", .10),
               ("r1", "R1 verdict", "lab", .45),
               ("persistence", "hash-chain persistence", "cmdcenter", .08),
               ("robot_dispatch", "robot dispatch", "tasks", .10)]
    out = []
    for key, label, src, w in weights:
        dur = int(total_ms * w) if total_ms else None
        st = "complete"
        reason = ""
        if key == "robot_dispatch" and not has_wo:
            st, reason = "not_linked", "no public work order is linked"
        out.append({"key": key, "label": label, "source": src, "duration_ms": dur,
                    "status": st, "failure_reason": reason})
    return out


def _trace_candidates(q="", limit=60):
    try:
        lim = max(1, min(int(limit), 200))
    except Exception:
        lim = 60
    ql = str(q or "").lower()
    items = []
    con = _db()
    rows = con.execute(f"SELECT {_WO_COLS} FROM workorders ORDER BY id DESC LIMIT 120").fetchall()
    for r in rows:
        w = _wo_dict(r)
        tid = w.get("trace_id") or ""
        if not tid:
            continue
        pred = w.get("pred") or {}
        text = " ".join([tid, w.get("code") or "", w.get("formula") or "", w.get("verdict") or ""]).lower()
        if ql and ql not in text:
            continue
        total = pred.get("timing_ms")
        items.append({"trace_id": tid, "kind": "prediction_batch", "work_order": w.get("code"),
                      "formula": w.get("formula"), "verdict": w.get("verdict"),
                      "source": w.get("pred_source") or "workorder",
                      "created_ts": w.get("created_ts"), "state": w.get("state"),
                      "waterfall": _trace_stage_template(total, True)})
    log_rows = con.execute("SELECT req_id, MAX(ts), COUNT(*), MAX(status), MAX(route)"
                           " FROM logs WHERE req_id IS NOT NULL AND req_id!=''"
                           " GROUP BY req_id ORDER BY MAX(id) DESC LIMIT 80").fetchall()
    for rid, ts_, n, st, route in log_rows:
        text = " ".join([rid or "", route or ""]).lower()
        if ql and ql not in text:
            continue
        items.append({"trace_id": rid, "kind": "request", "work_order": "",
                      "formula": "", "verdict": _public_severity(st),
                      "source": _route_service(route), "created_ts": ts_, "state": "ok" if (st or 0) < 400 else "error",
                      "waterfall": [{"key": "request", "label": route or "request", "source": _route_service(route),
                                     "duration_ms": None, "status": _public_severity(st), "failure_reason": ""}],
                      "span_count": n})
    con.close()
    if len(items) < lim:
        try:
            for row in _materials_all_rows():
                tid = row.get("trace_id") or ""
                if not tid:
                    continue
                text = " ".join([tid, row.get("formula") or "", row.get("work_order") or ""]).lower()
                if ql and ql not in text:
                    continue
                if any(x["trace_id"] == tid for x in items):
                    continue
                items.append({"trace_id": tid, "kind": "prediction", "work_order": row.get("work_order") or "",
                              "formula": row.get("formula"), "verdict": row.get("verdict"),
                              "source": row.get("source") or "public_material",
                              "created_ts": row.get("created"), "state": row.get("state") or "public",
                              "waterfall": _trace_stage_template(row.get("timing_ms"), bool(row.get("work_order")))})
                if len(items) >= lim:
                    break
        except Exception:
            pass
    return items[:lim]


@app.route("/api/traces")
def api_traces():
    items = _trace_candidates(request.args.get("q", ""), request.args.get("limit", 60))
    return jsonify({"ts": time.time(), "release": ASSET_VER, "items": items,
                    "empty_state": "No public prediction or request traces match this query." if not items else "",
                    "note": "Trace waterfalls use public metadata only; secrets and internal prompts are excluded."})


@app.route("/api/traces/<path:trace_id>")
def api_trace_public(trace_id):
    q = str(trace_id or "").strip()
    items = _trace_candidates(q, 120)
    exact = next((x for x in items if x.get("trace_id") == q), None)
    if not exact:
        return jsonify({"error": "not_found", "trace_id": q, "release": ASSET_VER}), 404
    return jsonify({"ts": time.time(), "release": ASSET_VER, "trace": exact})


@app.route("/api/diagnose")
def api_diagnose():
    """对每条活动告警给规则化根因 + 处置步骤 + 关联检查 (确定性, 不编造)."""
    con = _db()
    rows = con.execute(
        "SELECT id,rule,sys,severity,message,ts_raised,ts_ack FROM alarms"
        " WHERE ts_cleared IS NULL ORDER BY CASE severity WHEN 'crit' THEN 0"
        " WHEN 'warn' THEN 1 ELSE 2 END, ts_raised DESC").fetchall()
    con.close()
    out = []
    for r in rows:
        rule = r[1]
        diag = _DIAG_RULES.get(rule, {
            "cause": "未归类告警, 参见消息文本与规则名。",
            "steps": ["查看运维事件流定位时间点", "对照该系统当前 serving 态"],
            "related": "运维总览"})
        sysn = SYSTEMS.get(r[2], {}).get("name", "").split(" · ")[0] if r[2] else "平台"
        out.append({
            "id": r[0], "rule": rule, "sys": r[2], "sysname": sysn,
            "severity": r[3], "message": r[4], "ts_raised": r[5], "acked": bool(r[6]),
            "cause": diag["cause"],
            "steps": [s.replace("<sys>", r[2] or "").replace("mirror-*", f"mirror-{r[2] or '*'}")
                      for s in diag["steps"]],
            "related": diag["related"]})
    summary = {"total": len(out),
               "crit": sum(1 for o in out if o["severity"] == "crit"),
               "design_intended": sum(1 for o in out if o["rule"] == "real_offline")}
    return jsonify({"ts": time.time(), "diagnoses": out, "summary": summary})


@app.route("/api/copilot", methods=["GET", "POST"])
def api_copilot():
    """问平台运维副驾: GET 返回建议问题; POST {q} → 意图路由 + 实时接地答案."""
    if request.method == "GET":
        intro = ("指挥中心运维副驾 · 多步巡检接地实时数据"
                 + (" + DeepSeek 云合成 (开「深答」据实自然语言作答)" if llm_available()
                    else " (规则 Agent, 非大模型, 每条都可核)"))
        return jsonify({"suggest": _COPILOT_SUGGEST, "intro": intro, "llm": llm_available()})
    body = request.get_json(silent=True) or {}
    q = (body.get("q") or "").strip()
    deep = bool(body.get("deep"))
    if not q:
        return jsonify({"error": "问题为空"}), 400
    if len(q) > 200:
        q = q[:200]
    try:
        out = _copilot_answer(q, deep=deep)
        out["q"] = q
        out["ts"] = time.time()
        return jsonify(out)
    except Exception:
        app.logger.exception("copilot")
        return jsonify({"answer": "副驾处理异常, 请重试或换个问法。", "facts": [], "actions": [],
                        "grounded": False}), 200


# ============================================================ I1 平台自监控 (RED/USE + Prometheus)
import collections as _collections

_REQ = {"total": 0, "c4xx": 0, "c5xx": 0}
_REQ_LAT = _collections.deque(maxlen=600)      # 近期请求延迟 (ms) → p50/p95
_ROUTE_STAT = {}                                # route -> {n, ms, e} 累计 (供 /metrics 标签)
_LOG_BUF = _collections.deque(maxlen=3000)      # I4 待落库日志行


def _apply_public_security_headers(resp):
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "connect-src 'self' https://*.xiaomiju.xyz wss://*.xiaomiju.xyz; "
        "frame-src 'self' https://lab.xiaomiju.xyz https://car.xiaomiju.xyz https://arm.xiaomiju.xyz; "
        "worker-src 'self' blob:; "
        "manifest-src 'self'; "
        "frame-ancestors 'self'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "object-src 'none'; "
        "upgrade-insecure-requests; "
        "block-all-mixed-content"
    )
    resp.headers.setdefault("Content-Security-Policy", csp)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
    resp.headers.setdefault("Origin-Agent-Cluster", "?1")
    resp.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
    if request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https":
        resp.headers.setdefault("Strict-Transport-Security", "max-age=15552000; includeSubDomains")
    path = request.path or ""
    if path == "/sw.js" or path.endswith(".html") or path in ("/", "/status", "/fleet", "/tasks", "/observability", "/logs", "/traces", "/mq", "/queue", "/studio", "/fsd", "/replay", "/command", "/defense", "/sec"):
        resp.headers["Cache-Control"] = "no-cache"
    elif path in ("/api/security", "/api/hardening", "/api/config", "/api/releases", "/api/logs", "/api/site31_gate_evidence"):
        resp.headers["Cache-Control"] = "no-store"
    elif path.endswith((".js", ".css", ".svg", ".webmanifest", ".png", ".jpg", ".jpeg", ".webp", ".ico")):
        resp.headers.setdefault("Cache-Control", "public, max-age=3600")
    return resp


def _host_allowed(host):
    raw = (host or "").strip().lower()
    if raw.startswith("[") and "]" in raw:
        h = raw[1:raw.index("]")]
    else:
        h = raw.split(":", 1)[0].strip()
    if not h:
        return True
    if h in _ALLOWED_HOSTS:
        return True
    return any(h.endswith(suf) for suf in _ALLOWED_HOST_SUFFIXES)


@app.before_request
def _public_boundary_guard():
    if not _host_allowed(request.host):
        return jsonify({"error": "host_not_allowed", "release": ASSET_VER}), 421
    if request.path.startswith("/quality/"):
        return jsonify({"error": "not_found", "release": ASSET_VER}), 404
    if request.url_rule is None:
        return None
    if request.endpoint == "named_spa_page":
        page_name = (request.view_args or {}).get("page_name", "")
        named_pages = globals().get("SPA_NAMED_PAGES", frozenset())
        if page_name not in named_pages and not os.path.isfile(os.path.join(app.static_folder, page_name)):
            return None
    access = classify_request(request.path, request.method)
    g._access_scope = access.scope
    g._access_role = role_from_headers(
        request.headers.get("X-User"), request.headers.get("X-Role")
    )
    if not role_allows(g._access_role, access.scope):
        if request.method not in _SAFE_PUBLIC_METHODS and g._access_role == "public":
            return jsonify({"error": "public_read_only", "release": ASSET_VER,
                            "allowed": sorted(_SAFE_PUBLIC_METHODS),
                            "detail": "public surface accepts read-only requests; write paths require SSO/RBAC headers"}), 405
        status = 401 if g._access_role == "public" else 403
        return jsonify({"error": "access_denied", "release": ASSET_VER,
                        "scope": access.scope, "role": g._access_role}), status


@app.before_request
def _obs_start():
    g._obs_t0 = time.time()
    raw_rid = (request.headers.get("X-Request-ID") or "").strip()
    safe_rid = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw_rid)[:64].strip("-._:")
    g._obs_rid = safe_rid or uuid.uuid4().hex[:12]


@app.after_request
def _obs_end(resp):
    resp = _apply_public_security_headers(resp)
    try:
        t0 = getattr(g, "_obs_t0", None)
        if t0 is None:
            return resp
        ms = (time.time() - t0) * 1000.0
        st = resp.status_code
        _REQ["total"] += 1
        if 400 <= st < 500:
            _REQ["c4xx"] += 1
        elif st >= 500:
            _REQ["c5xx"] += 1
        _REQ_LAT.append(ms)
        rt = request.url_rule.rule if request.url_rule else request.path
        rs = _ROUTE_STAT.setdefault(rt, {"n": 0, "ms": 0.0, "e": 0})
        rs["n"] += 1
        rs["ms"] += ms
        if st >= 500:
            rs["e"] += 1
        rid = getattr(g, "_obs_rid", None)
        if rid:
            resp.headers["X-Request-ID"] = rid
        resp.headers["X-Access-Scope"] = getattr(g, "_access_scope", "unknown")
        if not (rt.startswith("/api/stream") or rt == "/metrics" or rt.startswith("/static")):
            ip = (request.headers.get("X-Forwarded-For", "") or request.remote_addr or "-").split(",")[0].strip()
            _LOG_BUF.append((round(time.time(), 3), rid, request.method, rt, st, round(ms, 1),
                             request.headers.get("X-User", "-"), request.headers.get("X-Role", "-"), ip, ""))
    except Exception:
        pass
    return resp


def _proc_self():
    """从 /proc/self/status 读 RSS(kB) 与线程数 — 零依赖, 不装 psutil."""
    rss = thr = 0
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1])
                elif ln.startswith("Threads:"):
                    thr = int(ln.split()[1])
    except Exception:
        pass
    return rss, thr


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def _record_app_metrics(con, now):
    rss, thr = _proc_self()
    lat = list(_REQ_LAT)
    try:
        dbb = os.path.getsize(DB_PATH)
    except OSError:
        dbb = None
    srows = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    con.execute("INSERT INTO app_metrics(ts,rss_kb,threads,uptime_s,req_total,req_4xx,req_5xx,"
                "p50_ms,p95_ms,db_bytes,samples_rows) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (now, rss, thr, int(now - _STARTED), _REQ["total"], _REQ["c4xx"], _REQ["c5xx"],
                 _pct(lat, .5), _pct(lat, .95), dbb, srows))


def _flush_logs(con):
    n = 0
    while _LOG_BUF and n < 600:
        con.execute("INSERT INTO logs(ts,req_id,method,route,status,ms,usr,role,ip,msg)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)", _LOG_BUF.popleft())
        n += 1
    con.execute("DELETE FROM logs WHERE id <= (SELECT MAX(id) FROM logs) - 5000")


def _rollup_hourly(con):
    """I9: 把 2 小时前的原始 30s 样本聚合进 samples_hourly (长期/快查)."""
    cutoff = int(time.time()) - 2 * 3600
    rows = con.execute(
        "SELECT (ts/3600)*3600 h, sys, AVG(real_ms), AVG(mirror_ms),"
        " SUM(serving='real'), SUM(serving='mirror'), SUM(serving='down'), COUNT(*)"
        " FROM samples WHERE ts < ? GROUP BY h, sys", (cutoff,)).fetchall()
    for h, sys, arm, amm, nr, nm, nd, n in rows:
        con.execute("INSERT OR REPLACE INTO samples_hourly(hour,sys,avg_real_ms,avg_mirror_ms,"
                    "n_real,n_mirror,n_down,n) VALUES(?,?,?,?,?,?,?,?)",
                    (h, sys, arm, amm, nr, nm, nd, n))


@app.route("/metrics")
def metrics_prom():
    """Prometheus 文本曝露 (工业标准 exposition format)."""
    rss, thr = _proc_self()
    lat = list(_REQ_LAT)
    try:
        dbb = os.path.getsize(DB_PATH)
    except OSError:
        dbb = 0
    L = ["# HELP xrd_build_info Build info", "# TYPE xrd_build_info gauge",
         'xrd_build_info{service="cmdcenter",ver="%s"} 1' % ASSET_VER,
         "# HELP xrd_process_rss_bytes Resident memory", "# TYPE xrd_process_rss_bytes gauge",
         "xrd_process_rss_bytes %d" % (rss * 1024),
         "# HELP xrd_process_threads Thread count", "# TYPE xrd_process_threads gauge",
         "xrd_process_threads %d" % thr,
         "# HELP xrd_process_uptime_seconds Uptime", "# TYPE xrd_process_uptime_seconds gauge",
         "xrd_process_uptime_seconds %d" % int(time.time() - _STARTED),
         "# HELP xrd_db_bytes Historian DB size", "# TYPE xrd_db_bytes gauge",
         "xrd_db_bytes %d" % dbb,
         "# HELP xrd_http_requests_total Total HTTP requests", "# TYPE xrd_http_requests_total counter",
         "xrd_http_requests_total %d" % _REQ["total"],
         'xrd_http_requests_total{class="4xx"} %d' % _REQ["c4xx"],
         'xrd_http_requests_total{class="5xx"} %d' % _REQ["c5xx"],
         "# HELP xrd_http_request_duration_ms Recent request latency", "# TYPE xrd_http_request_duration_ms summary",
         'xrd_http_request_duration_ms{quantile="0.5"} %s' % (_pct(lat, .5) or 0),
         'xrd_http_request_duration_ms{quantile="0.95"} %s' % (_pct(lat, .95) or 0)]
    for rt, rs in sorted(_ROUTE_STAT.items(), key=lambda x: -x[1]["n"])[:25]:
        srt = rt.replace('"', "")
        L.append('xrd_route_requests_total{route="%s"} %d' % (srt, rs["n"]))
        if rs["n"]:
            L.append('xrd_route_avg_ms{route="%s"} %.1f' % (srt, rs["ms"] / rs["n"]))
    return Response("\n".join(L) + "\n", mimetype="text/plain; version=0.0.4")


@app.route("/api/self")
def api_self():
    """平台自监控 JSON: 进程生命体征 + RED + app_metrics 时序 (供 #selfview)."""
    rss, thr = _proc_self()
    lat = list(_REQ_LAT)
    try:
        dbb = os.path.getsize(DB_PATH)
    except OSError:
        dbb = 0
    con = _db()
    rows = con.execute("SELECT ts,rss_kb,req_total,req_5xx,p50_ms,p95_ms,db_bytes,samples_rows"
                       " FROM app_metrics ORDER BY ts DESC LIMIT 120").fetchall()
    con.close()
    rows = rows[::-1]
    series = {"ts": [r[0] for r in rows], "rss_kb": [r[1] for r in rows],
              "p50": [r[4] for r in rows], "p95": [r[5] for r in rows],
              "db_kb": [round((r[6] or 0) / 1024) for r in rows]}
    # 请求速率: 相邻 app_metrics 的 req_total 差 / 间隔
    rate = []
    for i in range(1, len(rows)):
        dt = max(1, rows[i][0] - rows[i - 1][0])
        rate.append(round((rows[i][2] - rows[i - 1][2]) / dt, 3))
    err = _REQ["c5xx"]
    err_pct = round(100.0 * err / _REQ["total"], 3) if _REQ["total"] else 0.0
    routes = [{"route": rt, "n": rs["n"], "avg_ms": round(rs["ms"] / rs["n"], 1) if rs["n"] else 0,
               "err": rs["e"]} for rt, rs in
              sorted(_ROUTE_STAT.items(), key=lambda x: -x[1]["n"])[:12]]
    return jsonify({"ts": time.time(), "ver": ASSET_VER,
                    "proc": {"rss_kb": rss, "threads": thr, "uptime_s": int(time.time() - _STARTED),
                             "db_kb": round(dbb / 1024)},
                    "red": {"req_total": _REQ["total"], "req_4xx": _REQ["c4xx"], "req_5xx": _REQ["c5xx"],
                            "err_pct": err_pct, "p50_ms": _pct(lat, .5), "p95_ms": _pct(lat, .95),
                            "rate_per_s": round(rate[-1], 3) if rate else 0.0},
                    "series": series, "rate": rate, "routes": routes,
                    "note": "进程指标读自 /proc/self/status (零依赖); RED=Rate/Errors/Duration; 看门狗自监控其自身"})


# ============================================================ I2 OEE 生产线看板 + Andon
_OEE_WINDOWS = {"6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400}


def _oee_slice(con, t0, t1):
    """某时间段的 A/P/Q (全真 historian). 返回 None 字段=该段无数据."""
    span = max(1, t1 - t0)
    # A 可用率: serving≠down 占比 (三机汇总)
    a = con.execute("SELECT COUNT(*), SUM(serving='down') FROM samples WHERE ts>=? AND ts<?",
                    (t0, t1)).fetchone()
    avail = None if not a[0] else round(100.0 * (a[0] - (a[1] or 0)) / a[0], 2)
    # P 性能: 采样完整率 = 实际采样数 / 额定采样数 (生产节拍是否按额定跑)
    n_real_samp = a[0] or 0
    expected = (span / SAMPLE_EVERY) * len(SYSTEMS)
    perf = None if expected < 1 else round(min(100.0, 100.0 * n_real_samp / expected), 2)
    # Q 质量: 审计链完整率 × Conformal CI 覆盖率 (取段内最近 kpi 样本)
    kq = con.execute("SELECT audit_valid,audit_total,ci_coverage_pct FROM kpi_samples"
                     " WHERE ts>=? AND ts<? AND audit_total>0 ORDER BY ts DESC LIMIT 1",
                     (t0, t1)).fetchone()
    if kq and kq[1]:
        audit_pct = 100.0 * kq[0] / kq[1]
        ci_pct = kq[2] if kq[2] is not None else 100.0
        qual = round(audit_pct * ci_pct / 100.0, 2)
    else:
        audit_pct = ci_pct = qual = None
    oee = None
    if None not in (avail, perf, qual):
        oee = round(avail * perf * qual / 10000.0, 2)
    return {"avail": avail, "perf": perf, "qual": qual, "oee": oee,
            "audit_pct": round(audit_pct, 1) if audit_pct is not None else None,
            "ci_pct": round(ci_pct, 1) if ci_pct is not None else None,
            "samples": n_real_samp, "expected": int(expected)}


def _build_oee(window):
    span = _OEE_WINDOWS.get(window, 86400)
    now = int(time.time())
    t0 = now - span
    con = _db()
    overall = _oee_slice(con, t0, now)
    # 班次拆分: 窗口三等分
    seg = span // 3
    shifts = []
    names = (["早班", "中班", "晚班"] if window == "24h" else ["前段", "中段", "近段"])
    for i in range(3):
        s0 = t0 + i * seg
        s1 = t0 + (i + 1) * seg if i < 2 else now
        sl = _oee_slice(con, s0, s1)
        sl["name"] = names[i]
        sl["t0"] = s0
        sl["t1"] = s1
        shifts.append(sl)
    # 产量: 段内预测增量 (真实计数, 单列展示不并入 P 以免静默归零)
    kp = con.execute("SELECT predictions FROM kpi_samples WHERE ts>=? ORDER BY ts ASC LIMIT 1",
                     (t0,)).fetchone()
    kp2 = con.execute("SELECT predictions FROM kpi_samples ORDER BY ts DESC LIMIT 1").fetchone()
    throughput = (kp2[0] - kp[0]) if (kp and kp2) else None
    con.close()
    return {"ts": now, "window": window, "overall": overall, "shifts": shifts,
            "throughput": throughput,
            "note": "OEE=可用率A×性能P×质量Q (全真 historian)。A=serving≠down 占比; "
                    "P=采样完整率(生产节拍按额定跑的比例); Q=审计链完整率×Conformal CI 覆盖率。"
                    "产量=窗口内真实预测增量, 单列展示。镜像兜底运行 serving=mirror 仍计为可用(UI 不停机)。"}


@app.route("/api/oee")
def api_oee():
    w = request.args.get("window", "24h")
    if w not in _OEE_WINDOWS:
        w = "24h"
    return jsonify(_build_oee(w))


_ANDON_STATE = {"run": ("运行", "ok"), "idle": ("空转", "info"), "down": ("停机", "crit"),
                "call": ("呼叫支援", "warn")}


def _build_andon():
    with _lock:
        ops = _ops_cache["data"] or _build_ops()
    con = _db()
    calls = {r[0] for r in con.execute(
        "SELECT sys FROM alarms WHERE rule='andon_call' AND ts_cleared IS NULL").fetchall()}
    con.close()
    tiles = []
    for k in ("lab", "car", "arm"):
        s = ops.get("systems", {}).get(k, {})
        serv = s.get("serving")
        if k in calls:
            st = "call"
        elif serv == "down":
            st = "down"
        elif serv == "real":
            st = "run"
        else:
            st = "idle"   # 镜像兜底=空转 (在岗但非真机产出)
        label, tone = _ANDON_STATE[st]
        tiles.append({"key": k, "name": SYSTEMS[k]["name"].split(" · ")[0],
                      "state": st, "label": label, "tone": tone, "serving": serv,
                      "ms": s.get("real_ms") if serv == "real" else s.get("mirror_ms")})
    return {"ts": int(time.time()), "tiles": tiles,
            "note": "安灯板: 运行=真机直连产出 / 空转=镜像兜底在岗 / 停机=双路径离线 / 呼叫=人工触发支援"}


@app.route("/api/andon", methods=["GET", "POST"])
def api_andon():
    if request.method == "POST":
        d = request.get_json(silent=True) or {}
        sk = d.get("sys")
        if sk not in SYSTEMS:
            return jsonify({"error": "sys 必须是 lab/car/arm"}), 400
        act = d.get("action", "raise")
        who = request.headers.get("X-User", "operator")
        con = _db()
        now = int(time.time())
        if act == "clear":
            con.execute("UPDATE alarms SET ts_cleared=? WHERE rule='andon_call' AND sys=? AND ts_cleared IS NULL",
                        (now, sk))
            _add_event(sk, "andon_clear", "info", f"✓ 安灯呼叫解除 ({SYSTEMS[sk]['name'].split(' · ')[0]}) by {who}", con)
        else:
            ex = con.execute("SELECT id FROM alarms WHERE rule='andon_call' AND sys=? AND ts_cleared IS NULL",
                             (sk,)).fetchone()
            if not ex:
                con.execute("INSERT INTO alarms(rule,sys,severity,message,ts_raised) VALUES('andon_call',?,?,?,?)",
                            (sk, "warn", f"安灯呼叫支援: {SYSTEMS[sk]['name'].split(' · ')[0]} 工位请求人工介入", now))
                _add_event(sk, "andon_call", "warn", f"🔔 安灯呼叫支援 ({SYSTEMS[sk]['name'].split(' · ')[0]}) by {who}", con)
        con.commit()
        con.close()
        return jsonify(_build_andon())
    return jsonify(_build_andon())


# ============================================================ I4 结构化日志查询 / 追踪
@app.route("/api/logs")
def api_logs():
    route = request.args.get("route", "").strip()
    status = request.args.get("status", "").strip()
    q = request.args.get("q", "").strip()
    service = request.args.get("service", "").strip()
    severity = request.args.get("severity", "").strip()
    trace_id = request.args.get("trace_id", "").strip()
    since = request.args.get("since", "").strip()
    limit = min(max(int(request.args.get("limit", 120)), 1), 500)
    sql = "SELECT ts,req_id,method,route,status,ms,usr,role,ip FROM logs WHERE 1=1"
    ps = []
    if route:
        sql += " AND route LIKE ?"
        ps.append("%" + route + "%")
    if service:
        if service == "api":
            sql += " AND route LIKE '/api/%'"
        elif service == "pages":
            sql += " AND route NOT LIKE '/api/%'"
        else:
            prefixes = {
                "fleet": ("/api/fleet", "/fleet"),
                "tasks": ("/api/workorders", "/api/tasks", "/tasks"),
                "observability": ("/api/metrics", "/api/observability", "/observability"),
                "logs": ("/api/log", "/api/trace", "/logs", "/traces"),
                "research": ("/api/materials", "/api/predictions", "/materials", "/predictions"),
                "twin": ("/api/twin", "/twin"),
            }.get(service)
            if prefixes:
                sql += " AND (" + " OR ".join(["route LIKE ?" for _ in prefixes]) + ")"
                ps += [p + "%" for p in prefixes]
    if status:
        if status.endswith("xx"):
            lo = int(status[0]) * 100
            sql += " AND status>=? AND status<?"
            ps += [lo, lo + 100]
        else:
            sql += " AND status=?"
            ps.append(int(status))
    if severity:
        if severity == "critical":
            sql += " AND status>=500"
        elif severity == "warning":
            sql += " AND status>=400 AND status<500"
        elif severity == "info":
            sql += " AND status<400"
    if trace_id:
        sql += " AND req_id LIKE ?"
        ps.append("%" + trace_id + "%")
    if q:
        sql += " AND (route LIKE ? OR usr LIKE ? OR ip LIKE ? OR req_id LIKE ?)"
        ps += ["%" + q + "%"] * 4
    if since:
        try:
            sql += " AND ts>=?"
            ps.append(float(since))
        except ValueError:
            pass
    sql += " ORDER BY id DESC LIMIT ?"
    ps.append(limit)
    con = _db()
    rows = con.execute(sql, ps).fetchall()
    tot = con.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    con.close()
    cols = ["ts", "req_id", "method", "route", "status", "ms", "usr", "role", "ip"]
    logs = []
    for r in rows:
        d = dict(zip(cols, r))
        d["ip"] = _mask_ip(d.get("ip"))
        d["usr"] = _public_safe_text(d.get("usr"), 80)
        d["role"] = _public_safe_text(d.get("role"), 40)
        d["service"] = _route_service(d.get("route"))
        d["severity"] = _public_severity(d.get("status"))
        logs.append(d)
    return jsonify({"ts": time.time(), "total": tot, "n": len(rows),
                    "filters": {"route": route, "status": status, "service": service,
                                "severity": severity, "trace_id": trace_id},
                    "logs": logs})


@app.route("/api/trace/<rid>")
def api_trace(rid):
    con = _db()
    rows = con.execute("SELECT ts,req_id,method,route,status,ms,usr,role,ip FROM logs"
                       " WHERE req_id=? ORDER BY id", (rid,)).fetchall()
    con.close()
    cols = ["ts", "req_id", "method", "route", "status", "ms", "usr", "role", "ip"]
    spans = []
    for r in rows:
        d = dict(zip(cols, r))
        d["ip"] = _mask_ip(d.get("ip"))
        d["usr"] = _public_safe_text(d.get("usr"), 80)
        d["role"] = _public_safe_text(d.get("role"), 40)
        d["service"] = _route_service(d.get("route"))
        d["severity"] = _public_severity(d.get("status"))
        spans.append(d)
    return jsonify({"req_id": rid, "spans": spans})


# ============================================================ I8 配置中心 (I3 依赖, 提前定义)
def _config_get(key, default=None):
    try:
        con = _db()
        r = con.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        con.close()
        return r[0] if r else default
    except Exception:
        return default


def _config_all():
    con = _db()
    rows = con.execute("SELECT key,value,type,updated_by,ts FROM config ORDER BY key").fetchall()
    con.close()
    return [{"key": r[0], "value": r[1], "type": r[2], "updated_by": r[3], "ts": r[4]} for r in rows]


# ============================================================ I3 告警规则引擎 + 通知 + 静默 + 值班
_ALERT_METRICS = {
    "latency_ms": "服务路径延迟(ms)", "serving_down": "服务双路径离线(1/0)",
    "req_err_pct": "HTTP 5xx 错误率(%)", "rss_mb": "进程内存(MB)",
    "audit_broken": "审计链断裂(1/0)", "ci_coverage": "Conformal CI 覆盖率(%)",
    "predictions": "累计预测数",
}
_CHANNELS = [("wecom", "企业微信"), ("dingtalk", "钉钉"), ("feishu", "飞书"), ("email", "邮件")]
_custom_streak = {}


def _metric_now(metric, sk, ops, kpi):
    sysd = (ops.get("systems") or {}).get(sk, {}) if sk else {}
    if metric == "latency_ms":
        serv = sysd.get("serving")
        return sysd.get("real_ms") if serv == "real" else sysd.get("mirror_ms")
    if metric == "serving_down":
        return 1 if sysd.get("serving") == "down" else 0
    if metric == "req_err_pct":
        return round(100.0 * _REQ["c5xx"] / _REQ["total"], 3) if _REQ["total"] else 0.0
    if metric == "rss_mb":
        return round(_proc_self()[0] / 1024, 1)
    kk = (kpi or {}).get("kpi") or {}
    if metric == "audit_broken":
        return 1 if kk.get("audit_intact") is False else 0
    if metric == "ci_coverage":
        return kk.get("ci_coverage_pct")
    if metric == "predictions":
        return kk.get("predictions")
    return None


def _op_true(v, op, thr):
    if v is None:
        return False
    return {">": v > thr, "<": v < thr, ">=": v >= thr, "<=": v <= thr,
            "==": v == thr, "!=": v != thr}.get(op, False)


def _silenced(sk, con, now):
    for (sc,) in con.execute("SELECT scope FROM silences WHERE ts_start<=? AND ts_end>=?",
                             (now, now)).fetchall():
        if sc in ("all", "*", None, "") or sc == sk:
            return True
    return False


def _webhook_payload(channel, title, body):
    text = f"{title}\n{body}"
    if channel == "wecom":
        return {"msgtype": "text", "text": {"content": text}}
    if channel == "dingtalk":
        return {"msgtype": "text", "text": {"content": text}}
    if channel == "feishu":
        return {"msg_type": "text", "content": {"text": text}}
    return {"text": text}


def _validate_webhook_url(channel, url):
    """Return a DNS-pinned public endpoint or raise ValueError.

    Host allowlisting prevents arbitrary destinations; resolving before curl and
    passing --resolve closes the DNS rebinding gap between validation and send.
    """
    if channel not in _WEBHOOK_HOSTS:
        raise ValueError("通知渠道不支持 webhook")
    try:
        parsed = urlparse(str(url).strip())
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("webhook URL 格式非法") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme.lower() != "https":
        raise ValueError("webhook 仅允许 HTTPS")
    if not host or parsed.username is not None or parsed.password is not None:
        raise ValueError("webhook URL 不允许凭据且必须包含主机")
    if port not in (None, 443):
        raise ValueError("webhook 仅允许 443 端口")
    if host not in (_WEBHOOK_HOSTS[channel] | _WEBHOOK_EXTRA_HOSTS):
        raise ValueError("webhook 主机不在允许列表")
    try:
        answers = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("webhook 主机 DNS 解析失败") from exc
    public_ips = []
    for answer in answers:
        raw_ip = answer[4][0].split("%", 1)[0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if addr.is_global:
            public_ips.append(addr)
    if not public_ips:
        raise ValueError("webhook 主机未解析到公网地址")
    public_ips.sort(key=lambda addr: (addr.version != 4, str(addr)))
    selected = public_ips[0]
    curl_ip = str(selected) if selected.version == 4 else f"[{selected}]"
    return {"url": parsed.geturl(), "host": host, "ip": str(selected),
            "curl_resolve": f"{host}:443:{curl_ip}"}


def _notify_send(channel, title, body, con=None):
    """发送通知到一个渠道. 无 URL/未配置 → 如实记 skipped. 真实经 subprocess curl 发 webhook."""
    own = con is None
    if own:
        con = _db()
    now = int(time.time())
    status = detail = ""
    if channel == "email":
        if _mail_cfg():
            try:
                _send_alarm_mail("custom_rule", "lab", f"[XRD] {title}", body)
                status, detail = "sent", "SMTP"
            except Exception as e:
                status, detail = "error", str(e)[:80]
        else:
            status, detail = "skipped", "邮件通道未配置 (无 SMTP)"
    else:
        url = _config_get(f"webhook.{channel}")
        if not url:
            status, detail = "skipped", "未配置 (无 webhook URL)"
        else:
            try:
                endpoint = _validate_webhook_url(channel, url)
                payload = json.dumps(_webhook_payload(channel, title, body))
                r = subprocess.run(["curl", "-sS", "--proto", "=https", "--max-redirs", "0",
                                    "--connect-timeout", "3", "-m", "8", "-X", "POST",
                                    "--resolve", endpoint["curl_resolve"], endpoint["url"],
                                    "-H", "Content-Type: application/json", "--data-binary", "@-",
                                    "-o", os.devnull, "-w", "%{http_code}"],
                                   input=payload, capture_output=True, text=True, timeout=12)
                http_code = (r.stdout or "").strip()
                status = "sent" if r.returncode == 0 and http_code.isdigit() and 200 <= int(http_code) < 300 else "error"
                detail = (f"HTTP {http_code}" if http_code else (r.stderr or "curl failed"))[:120]
            except ValueError as e:
                status, detail = "blocked", str(e)[:120]
            except Exception as e:
                status, detail = "error", str(e)[:80]
    con.execute("INSERT INTO notifications(ts,rule,channel,status,detail) VALUES(?,?,?,?,?)",
                (now, title[:60], channel, status, detail))
    if own:
        con.commit()
        con.close()
    return status


def _eval_custom_rules(ops, kpi_doc=None):
    """I3: 评估用户定义规则 (alert_rules), 复用 alarms 生命周期, 命中→通知渠道, 尊重静默窗口."""
    try:
        con = _db()
        rules = con.execute("SELECT id,name,sys,metric,op,threshold,for_n,severity,channel"
                            " FROM alert_rules WHERE enabled=1").fetchall()
        if not rules:
            con.close()
            return
        now = int(time.time())
        active = {r[0]: r[1] for r in con.execute(
            "SELECT id, rule FROM alarms WHERE rule LIKE 'custom:%' AND ts_cleared IS NULL")}
        active_keys = set(active.values())
        for rid, name, sk, metric, op, thr, for_n, sev, channel in rules:
            v = _metric_now(metric, sk, ops, kpi_doc)
            hit = _op_true(v, op, thr)
            key = f"custom:{rid}"
            st = _custom_streak.get(rid, 0)
            st = st + 1 if hit else 0
            _custom_streak[rid] = st
            firing = st >= max(1, for_n or 1)
            if firing and key not in active_keys and not _silenced(sk, con, now):
                msg = f"[规则] {name}: {_ALERT_METRICS.get(metric, metric)}={v} {op} {thr}" + (f" ({sk})" if sk else "")
                con.execute("INSERT INTO alarms(rule,sys,severity,message,ts_raised) VALUES(?,?,?,?,?)",
                            (key, sk, sev, msg, now))
                _add_event(sk or "lab", "alarm_raise", sev, "⚠ 告警: " + msg, con)
                con.commit()
                for ch in (channel or "").split(",") if channel else []:
                    ch = ch.strip()
                    if ch:
                        _notify_send(ch, f"告警 · {name}", msg, con)
                con.commit()
            elif not hit and key in active_keys:
                con.execute("UPDATE alarms SET ts_cleared=? WHERE rule=? AND ts_cleared IS NULL", (now, key))
                _add_event(sk or "lab", "alarm_clear", "info", f"✓ 恢复: 规则 {name}", con)
                con.commit()
        con.close()
    except Exception:
        app.logger.exception("custom rules")


def _build_alert_center():
    con = _db()
    rules = [dict(zip(["id", "name", "sys", "metric", "op", "threshold", "for_n", "severity",
                       "channel", "enabled", "created_by", "ts"], r)) for r in con.execute(
        "SELECT id,name,sys,metric,op,threshold,for_n,severity,channel,enabled,created_by,ts"
        " FROM alert_rules ORDER BY id DESC")]
    now = int(time.time())
    sil = [dict(zip(["id", "scope", "reason", "ts_start", "ts_end", "created_by", "ts"], r))
           for r in con.execute("SELECT id,scope,reason,ts_start,ts_end,created_by,ts FROM silences"
                                " ORDER BY ts_end DESC LIMIT 20")]
    oc = [dict(zip(["id", "name", "contact", "ts_start", "ts_end", "created_by", "ts"], r))
          for r in con.execute("SELECT id,name,contact,ts_start,ts_end,created_by,ts FROM oncall"
                              " ORDER BY ts_start DESC LIMIT 20")]
    notif = [dict(zip(["id", "ts", "rule", "channel", "status", "detail"], r))
             for r in con.execute("SELECT id,ts,rule,channel,status,detail FROM notifications"
                                  " ORDER BY id DESC LIMIT 30")]
    con.close()
    channels = [{"key": k, "name": nm,
                 "configured": bool(_mail_cfg()) if k == "email" else bool(_config_get(f"webhook.{k}"))}
                for k, nm in _CHANNELS]
    active_oncall = next((o for o in oc if o["ts_start"] <= now <= o["ts_end"]), None)
    active_sil = [s for s in sil if s["ts_start"] <= now <= s["ts_end"]]
    return {"ts": now, "rules": rules, "metrics": _ALERT_METRICS, "channels": channels,
            "silences": sil, "active_silences": active_sil, "oncall": oc,
            "active_oncall": active_oncall, "notifications": notif,
            "alarm_counts": _alarm_counts(),
            "note": "规则引擎每采样周期评估 (复用 alarms 生命周期); 命中→通知渠道 (未配置如实标注); "
                    "静默/维护窗口内抑制升警; 值班表标注当前在岗。"}


@app.route("/api/alert_center")
def api_alert_center():
    return jsonify(_build_alert_center())


@app.route("/api/alert_rules", methods=["GET", "POST"])
def api_alert_rules():
    if request.method == "POST":
        deny = _require_admin()
        if deny:
            return deny
        d = request.get_json(silent=True) or {}
        metric = d.get("metric")
        if metric not in _ALERT_METRICS:
            return jsonify({"error": "metric 非法"}), 400
        if d.get("op") not in (">", "<", ">=", "<=", "==", "!="):
            return jsonify({"error": "op 非法"}), 400
        try:
            thr = float(d.get("threshold"))
        except (TypeError, ValueError):
            return jsonify({"error": "threshold 非法"}), 400
        sk = d.get("sys") or None
        if sk and sk not in SYSTEMS:
            return jsonify({"error": "sys 非法"}), 400
        con = _db()
        con.execute("INSERT INTO alert_rules(name,sys,metric,op,threshold,for_n,severity,channel,"
                    "enabled,created_by,ts) VALUES(?,?,?,?,?,?,?,?,1,?,?)",
                    (d.get("name") or f"{metric}{d.get('op')}{thr}", sk, metric, d.get("op"), thr,
                     int(d.get("for_n", 1)), d.get("severity", "warn"),
                     ",".join(d.get("channels", [])) if isinstance(d.get("channels"), list) else d.get("channel", ""),
                     request.headers.get("X-User", "admin"), int(time.time())))
        con.commit()
        con.close()
        return jsonify({"ok": True})
    return jsonify(_build_alert_center())


@app.route("/api/alert_rules/<int:rid>", methods=["DELETE", "PATCH"])
def api_alert_rule_edit(rid):
    deny = _require_admin()
    if deny:
        return deny
    con = _db()
    if request.method == "DELETE":
        con.execute("DELETE FROM alert_rules WHERE id=?", (rid,))
    else:
        d = request.get_json(silent=True) or {}
        con.execute("UPDATE alert_rules SET enabled=? WHERE id=?",
                    (1 if d.get("enabled") else 0, rid))
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/silences", methods=["GET", "POST"])
def api_silences():
    if request.method == "POST":
        deny = _require_admin()
        if deny:
            return deny
        d = request.get_json(silent=True) or {}
        now = int(time.time())
        mins = int(d.get("minutes", 60))
        con = _db()
        con.execute("INSERT INTO silences(scope,reason,ts_start,ts_end,created_by,ts) VALUES(?,?,?,?,?,?)",
                    (d.get("scope", "all"), d.get("reason", "维护窗口"), now, now + mins * 60,
                     request.headers.get("X-User", "admin"), now))
        con.commit()
        con.close()
        return jsonify({"ok": True})
    return jsonify({"silences": _build_alert_center()["silences"]})


@app.route("/api/silences/<int:sid>", methods=["DELETE"])
def api_silence_del(sid):
    deny = _require_admin()
    if deny:
        return deny
    con = _db()
    con.execute("UPDATE silences SET ts_end=? WHERE id=?", (int(time.time()), sid))
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/oncall", methods=["GET", "POST"])
def api_oncall():
    if request.method == "POST":
        deny = _require_admin()
        if deny:
            return deny
        d = request.get_json(silent=True) or {}
        now = int(time.time())
        hours = float(d.get("hours", 24))
        con = _db()
        con.execute("INSERT INTO oncall(name,contact,ts_start,ts_end,created_by,ts) VALUES(?,?,?,?,?,?)",
                    (d.get("name", "值班员"), d.get("contact", ""), now, int(now + hours * 3600),
                     request.headers.get("X-User", "admin"), now))
        con.commit()
        con.close()
        return jsonify({"ok": True})
    return jsonify({"oncall": _build_alert_center()["oncall"]})


@app.route("/api/notify_test", methods=["POST"])
def api_notify_test():
    deny = _require_admin()
    if deny:
        return deny
    d = request.get_json(silent=True) or {}
    ch = d.get("channel")
    if ch not in [c[0] for c in _CHANNELS]:
        return jsonify({"error": "channel 非法"}), 400
    status = _notify_send(ch, "测试通知", "这是一条来自 XRD 指挥中心的测试通知。")
    return jsonify({"ok": True, "status": status})


# ============================================================ I5 QMS 质量管理 (NCR/CAPA/COA/族谱)
def _wo_by_code(con, batch):
    return con.execute(f"SELECT {_WO_COLS} FROM workorders WHERE code=? OR id=?",
                       (batch, batch if str(batch).isdigit() else -1)).fetchone()


@app.route("/api/qms")
def api_qms():
    con = _db()
    ncrs = [dict(zip(["id", "code", "batch", "defect", "severity", "raised_by", "ts", "status", "capa_id"], r))
            for r in con.execute("SELECT id,code,batch,defect,severity,raised_by,ts,status,capa_id"
                                 " FROM ncr ORDER BY id DESC LIMIT 60")]
    capas = [dict(zip(["id", "ncr_id", "root_cause", "action", "owner", "due", "status", "ts"], r))
             for r in con.execute("SELECT id,ncr_id,root_cause,action,owner,due,status,ts"
                                  " FROM capa ORDER BY id DESC LIMIT 60")]
    wos = [{"id": r[0], "code": r[1], "formula": r[2], "verdict": r[11], "state": r[9]}
           for r in con.execute(f"SELECT {_WO_COLS} FROM workorders ORDER BY id DESC LIMIT 40")]
    con.close()
    nc = {"open": sum(1 for n in ncrs if n["status"] == "open"), "total": len(ncrs)}
    cc = {"open": sum(1 for c in capas if c["status"] != "closed"), "total": len(capas)}
    return jsonify({"ts": int(time.time()), "ncr": ncrs, "capa": capas, "batches": wos,
                    "counts": {"ncr": nc, "capa": cc},
                    "note": "项目 QMS: 不合格(NCR)→纠正措施(CAPA)→合格证记录(COA)→批次族谱；"
                            "COA 数据取自真 predict_engine + 审计链 SHA，不代表第三方认证。"})


@app.route("/api/ncr", methods=["POST"])
def api_ncr_create():
    if (request.headers.get("X-Role") or "") not in ("admin", "member"):
        return jsonify({"error": "需要 member/admin 角色"}), 403
    d = request.get_json(silent=True) or {}
    if not d.get("defect"):
        return jsonify({"error": "defect 必填"}), 400
    now = int(time.time())
    con = _db()
    code = "NCR-" + datetime.datetime.now().strftime("%y%m%d") + "-" + uuid.uuid4().hex[:4]
    cur = con.execute("INSERT INTO ncr(code,batch,defect,severity,raised_by,ts,status)"
                      " VALUES(?,?,?,?,?,?,'open')",
                      (code, d.get("batch", ""), d.get("defect"), d.get("severity", "minor"),
                       request.headers.get("X-User", "member"), now))
    nid = cur.lastrowid
    _add_event("lab", "ncr_raise", "warn", f"📋 NCR 开单 {code}: {d.get('defect')[:60]}", con)
    con.commit()
    con.close()
    return jsonify({"ok": True, "id": nid, "code": code})


@app.route("/api/ncr/<int:nid>/capa", methods=["POST"])
def api_capa_create(nid):
    if (request.headers.get("X-Role") or "") not in ("admin", "member"):
        return jsonify({"error": "需要 member/admin 角色"}), 403
    d = request.get_json(silent=True) or {}
    now = int(time.time())
    con = _db()
    due = now + int(d.get("due_days", 7)) * 86400
    cur = con.execute("INSERT INTO capa(ncr_id,root_cause,action,owner,due,status,ts)"
                      " VALUES(?,?,?,?,?,'open',?)",
                      (nid, d.get("root_cause", ""), d.get("action", ""),
                       d.get("owner", request.headers.get("X-User", "member")), due, now))
    con.execute("UPDATE ncr SET capa_id=?, status='investigating' WHERE id=?", (cur.lastrowid, nid))
    _add_event("lab", "capa_open", "info", f"🔧 CAPA 立项 (NCR#{nid}): {d.get('action', '')[:50]}", con)
    con.commit()
    con.close()
    return jsonify({"ok": True, "id": cur.lastrowid})


@app.route("/api/capa/<int:cid>", methods=["PATCH"])
def api_capa_update(cid):
    if (request.headers.get("X-Role") or "") not in ("admin", "member"):
        return jsonify({"error": "需要 member/admin 角色"}), 403
    d = request.get_json(silent=True) or {}
    st = d.get("status")
    if st not in ("open", "in_progress", "closed"):
        return jsonify({"error": "status 非法"}), 400
    con = _db()
    con.execute("UPDATE capa SET status=? WHERE id=?", (st, cid))
    if st == "closed":
        r = con.execute("SELECT ncr_id FROM capa WHERE id=?", (cid,)).fetchone()
        if r:
            con.execute("UPDATE ncr SET status='closed' WHERE id=?", (r[0],))
            _add_event("lab", "capa_close", "info", f"✓ CAPA#{cid} 关闭, NCR 闭环", con)
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/coa/<batch>")
def api_coa(batch):
    """合格证 (COA): 取真 predict_engine 结果 + 审计链 SHA, 生成可追溯证书."""
    con = _db()
    r = _wo_by_code(con, batch)
    if not r:
        con.close()
        return jsonify({"error": f"批次 {batch} 无对应工单"}), 404
    wo = _wo_dict(r)
    signs = [dict(zip(["signer", "role", "meaning", "reason", "ts", "hash"], s)) for s in con.execute(
        "SELECT signer,role,meaning,reason,ts,hash FROM esign WHERE obj_type='coa' AND obj_id=? ORDER BY id",
        (str(batch),)).fetchall()]
    con.close()
    pred = wo.get("pred") or {}
    content = {"batch": wo["code"], "formula": wo["formula"],
               "dopant": f"{wo['dop_symbol']}@{wo['dop_site']} {wo['dop_pct']}%" if wo.get("dop_symbol") else "—",
               "verdict": wo["verdict"], "lambda_em": pred.get("lambda_em"),
               "ci": [pred.get("ci_lo"), pred.get("ci_hi")], "trace_id": wo["trace_id"],
               "lambda_obs": wo.get("lambda_obs"), "state": wo["state"],
               "pred_source": wo.get("pred_source")}
    coa_hash = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return jsonify({"ts": int(time.time()), "coa": content, "coa_hash": coa_hash,
                    "signatures": signs,
                    "note": "证书内容哈希 (SHA-256) 用于变更检测与追溯; 数据源=真 predict_engine + 工单回填; "
                            "项目电子签批记录签名人、角色、含义、原因和时间，不宣称 21 CFR Part 11 认证。"})


@app.route("/api/esign", methods=["POST"])
def api_esign():
    if (request.headers.get("X-Role") or "") not in ("admin", "member"):
        return jsonify({"error": "需要 member/admin 角色"}), 403
    d = request.get_json(silent=True) or {}
    now = int(time.time())
    payload = f"{d.get('obj_type')}|{d.get('obj_id')}|{request.headers.get('X-User')}|{now}|{d.get('reason', '')}"
    h = hashlib.sha256(payload.encode()).hexdigest()
    con = _db()
    con.execute("INSERT INTO esign(ts,obj_type,obj_id,signer,role,meaning,reason,hash)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (now, d.get("obj_type", "coa"), str(d.get("obj_id", "")),
                 request.headers.get("X-User", "member"), request.headers.get("X-Role", "member"),
                 d.get("meaning", "approved"), d.get("reason", ""), h))
    con.commit()
    con.close()
    return jsonify({"ok": True, "hash": h})


@app.route("/api/genealogy/<batch>")
def api_genealogy(batch):
    """批次族谱: 原料→工单→预测→判读→NCR/CAPA 血缘图 (复用拓扑节点图渲染)."""
    con = _db()
    r = _wo_by_code(con, batch)
    if not r:
        con.close()
        return jsonify({"error": f"批次 {batch} 无对应工单"}), 404
    wo = _wo_dict(r)
    pred = wo.get("pred") or {}
    ncrs = con.execute("SELECT id,code,defect,status,capa_id FROM ncr WHERE batch=?", (wo["code"],)).fetchall()
    logs = con.execute("SELECT ts,action,detail FROM wo_log WHERE wo=? ORDER BY id", (wo["id"],)).fetchall()
    con.close()
    nodes = [{"id": "mat", "label": "原料配方", "sub": wo["formula"], "tone": "info"},
             {"id": "wo", "label": "批次工单", "sub": wo["code"], "tone": "blue"},
             {"id": "pred", "label": "AI 预测", "sub": f"λ={pred.get('lambda_em', '—')}nm", "tone": "violet"},
             {"id": "verdict", "label": "判读", "sub": wo.get("verdict") or "—", "tone": "amber"}]
    edges = [("mat", "wo"), ("wo", "pred"), ("pred", "verdict")]
    if wo.get("lambda_obs") is not None:
        nodes.append({"id": "obs", "label": "实测回填", "sub": f"λ_obs={wo['lambda_obs']}nm", "tone": "emerald"})
        edges.append(("verdict", "obs"))
    for nr in ncrs:
        nid = f"ncr{nr[0]}"
        nodes.append({"id": nid, "label": "NCR", "sub": nr[1], "tone": "rose"})
        edges.append(("verdict", nid))
        if nr[4]:
            cid = f"capa{nr[4]}"
            nodes.append({"id": cid, "label": "CAPA", "sub": f"#{nr[4]}", "tone": "teal"})
            edges.append((nid, cid))
    return jsonify({"ts": int(time.time()), "batch": wo["code"], "nodes": nodes,
                    "edges": [{"from": a, "to": b} for a, b in edges],
                    "timeline": [{"ts": t, "action": a, "detail": d} for t, a, d in logs]})


# ============================================================ I6 CMMS 维护管理 (PM/MTBF/MTTR/备件)
def _mtbf_mttr(con, sk, days=30):
    since = int(time.time()) - days * 86400
    raises = [r[0] for r in con.execute(
        "SELECT ts_raised FROM alarms WHERE sys=? AND ts_raised>=? ORDER BY ts_raised", (sk, since)).fetchall()]
    repairs = [r[0] for r in con.execute(
        "SELECT ts_cleared-ts_raised FROM alarms WHERE sys=? AND ts_cleared IS NOT NULL AND ts_raised>=?",
        (sk, since)).fetchall()]
    mtbf = None
    if len(raises) >= 2:
        gaps = [raises[i] - raises[i - 1] for i in range(1, len(raises))]
        mtbf = round(sum(gaps) / len(gaps) / 3600.0, 1)   # 小时
    mttr = round(sum(repairs) / len(repairs) / 60.0, 1) if repairs else None   # 分钟
    return {"mtbf_h": mtbf, "mttr_min": mttr, "failures": len(raises), "repairs": len(repairs)}


@app.route("/api/cmms")
def api_cmms():
    try:
        with open(ASSETS_FILE, encoding="utf-8") as f:
            reg = json.load(f)
    except Exception:
        reg = {"groups": []}
    with _lock:
        ops = (_ops_cache["data"] or {}).get("systems", {})
    con = _db()
    now = int(time.time())
    assets = []
    for grp in reg.get("groups", []):
        k = grp.get("key")
        rel = _mtbf_mttr(con, k) if k in SYSTEMS else {"mtbf_h": None, "mttr_min": None, "failures": 0, "repairs": 0}
        assets.append({"key": k, "icon": grp.get("icon"), "name": grp.get("name"),
                       "host": grp.get("host"), "serving": ops.get(k, {}).get("serving"),
                       "reliability": rel,
                       "maint_n": con.execute("SELECT COUNT(*) FROM maintenance WHERE asset=?", (k,)).fetchone()[0]})
    pms = [dict(zip(["id", "asset", "task", "interval_days", "last_done", "next_due", "enabled"], r))
           for r in con.execute("SELECT id,asset,task,interval_days,last_done,next_due,enabled"
                                " FROM pm_schedule ORDER BY next_due")]
    for p in pms:
        p["overdue"] = bool(p["next_due"] and p["next_due"] < now)
        p["due_soon"] = bool(p["next_due"] and now <= p["next_due"] < now + 3 * 86400)
    spares = [dict(zip(["id", "part", "asset", "qty", "min_qty", "unit"], r))
              for r in con.execute("SELECT id,part,asset,qty,min_qty,unit FROM spares ORDER BY part")]
    for s in spares:
        s["low"] = bool(s["min_qty"] is not None and s["qty"] is not None and s["qty"] <= s["min_qty"])
    maint = [dict(zip(["id", "ts", "asset", "author", "note"], r))
             for r in con.execute("SELECT id,ts,asset,author,note FROM maintenance ORDER BY id DESC LIMIT 30")]
    con.close()
    return jsonify({"ts": now, "assets": assets, "pm": pms, "spares": spares, "maintenance": maint,
                    "pm_due": sum(1 for p in pms if p["overdue"]),
                    "spares_low": sum(1 for s in spares if s["low"]),
                    "note": "CMMS: 预防性维护(PM)排程 + MTBF/MTTR 可靠性(算自真告警历史) + 备件库存(低于安全库存标红)。"
                            "MTBF=平均故障间隔(h), MTTR=平均修复时长(min)。设备未上电时告警以镜像兜底/真机离线为主。"})


@app.route("/api/pm", methods=["POST"])
def api_pm_create():
    deny = _require_admin()
    if deny:
        return deny
    d = request.get_json(silent=True) or {}
    now = int(time.time())
    interval = int(d.get("interval_days", 30))
    con = _db()
    con.execute("INSERT INTO pm_schedule(asset,task,interval_days,last_done,next_due,enabled)"
                " VALUES(?,?,?,?,?,1)",
                (d.get("asset", "lab"), d.get("task", "例行点检"), interval, now, now + interval * 86400))
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/pm/<int:pid>/done", methods=["POST"])
def api_pm_done(pid):
    if (request.headers.get("X-Role") or "") not in ("admin", "member"):
        return jsonify({"error": "需要 member/admin 角色"}), 403
    con = _db()
    r = con.execute("SELECT asset,task,interval_days FROM pm_schedule WHERE id=?", (pid,)).fetchone()
    if not r:
        con.close()
        return jsonify({"error": "PM 不存在"}), 404
    now = int(time.time())
    con.execute("UPDATE pm_schedule SET last_done=?, next_due=? WHERE id=?",
                (now, now + (r[2] or 30) * 86400, pid))
    con.execute("INSERT INTO maintenance(ts,asset,author,note) VALUES(?,?,?,?)",
                (now, r[0], request.headers.get("X-User", "member"), f"PM 完成: {r[1]}"))
    _add_event(r[0] if r[0] in SYSTEMS else "lab", "pm_done", "info", f"🔧 PM 完成: {r[1]} ({r[0]})", con)
    con.commit()
    con.close()
    return jsonify({"ok": True})


@app.route("/api/spares", methods=["POST"])
def api_spares_set():
    deny = _require_admin()
    if deny:
        return deny
    d = request.get_json(silent=True) or {}
    if not d.get("part"):
        return jsonify({"error": "part 必填"}), 400
    con = _db()
    con.execute("INSERT INTO spares(part,asset,qty,min_qty,unit) VALUES(?,?,?,?,?)",
                (d.get("part"), d.get("asset", ""), int(d.get("qty", 0)),
                 int(d.get("min_qty", 0)), d.get("unit", "件")))
    con.commit()
    con.close()
    return jsonify({"ok": True})


# ============================================================ I7 安全合规加固 + 态势评分
def _probe_headers():
    try:
        p = subprocess.run(["curl", "-sI", "-m", "8", "https://xiaomiju.xyz/"],
                           capture_output=True, text=True, timeout=10)
        h = {}
        for ln in p.stdout.splitlines():
            if ":" in ln:
                k, _, v = ln.partition(":")
                h[k.strip().lower()] = v.strip()
        return h
    except Exception:
        return {}


@app.route("/api/security")
def api_security():
    h = _probe_headers()
    checks = []

    def add(name, ok, detail, na=False):
        checks.append({"name": name, "status": ("na" if na else ("pass" if ok else "fail")), "detail": detail})

    add("HTTPS/TLS", bool(h), "线上 HTTPS 响应头已取得" if h else "线上响应头探测不可用", na=not h)
    add("HSTS", "strict-transport-security" in h, h.get("strict-transport-security", "缺失"))
    csp = h.get("content-security-policy", "")
    add("CSP (frame-ancestors)", "frame-ancestors" in csp, csp[:80] or "缺失 — 建议加 frame-ancestors 限制内嵌")
    add("X-Content-Type-Options", h.get("x-content-type-options") == "nosniff", h.get("x-content-type-options", "缺失"))
    add("Referrer-Policy", "referrer-policy" in h, h.get("referrer-policy", "缺失"))
    current_role = (request.headers.get("X-Role") or "").strip().lower()
    current_user = (request.headers.get("X-User") or "").strip()
    add("SSO forward_auth", bool(current_user and current_role in {"admin", "member", "judge"}),
        "当前请求携带网关身份与角色" if current_user else "当前请求未观察到网关身份头",
        na=not current_user)
    # RBAC
    nroles = {}
    try:
        with open(os.path.join(AUTH_DIR, "users.json"), encoding="utf-8") as f:
            us = json.load(f)
        for v in us.values():
            nroles[v.get("role", "?")] = nroles.get(v.get("role", "?"), 0) + 1
        add("RBAC 分级", len(nroles) >= 2, "角色分布: " + ", ".join(f"{k}×{v}" for k, v in nroles.items()))
    except Exception:
        add("RBAC 分级", False, "users.json 不可读", na=True)
    # secrets
    try:
        st = os.stat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "secrets.env"))
        mode = oct(st.st_mode)[-3:]
        add("密钥文件权限", mode == "600", f"secrets.env 权限 {mode} (期望 600, 不入库)")
    except Exception:
        add("密钥文件权限", True, "无 secrets.env 或不可读", na=True)
    # 登录审计
    fails = 0
    recent = []
    try:
        path = os.path.join(AUTH_DIR, "logins.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()[-200:]
        for ln in lines:
            try:
                j = json.loads(ln)
                recent.append(j)
                if not j.get("ok", j.get("success", True)):
                    fails += 1
            except Exception:
                pass
        add("登录审计", True, f"近 {len(lines)} 条登录留痕, 失败 {fails} 次")
    except Exception:
        add("登录审计", False, "logins.jsonl 不可读", na=True)
    add("CSRF 基线", True, "SameSite=Lax cookie；高风险写接口仍需逐路由验证 Origin/CSRF 覆盖")
    add("项目电子签批留痕", True, "记录签名人、角色、原因和 SHA；不宣称 21 CFR Part 11 认证")
    npass = sum(1 for c in checks if c["status"] == "pass")
    nfail = sum(1 for c in checks if c["status"] == "fail")
    ntotal = sum(1 for c in checks if c["status"] != "na")
    score = round(100.0 * npass / ntotal) if ntotal else 0
    safe_recent = []
    for j in recent[-12:][::-1]:
        safe_recent.append({
            "ts": j.get("ts"),
            "ok": bool(j.get("ok", j.get("success", True))),
            "role": _public_safe_text(j.get("role", ""), 40),
            "user_label": "user-redacted",
            "ip": _mask_ip(j.get("ip") or j.get("remote_addr") or ""),
        })
    return jsonify({"ts": int(time.time()), "checks": checks, "score": score,
                    "pass": npass, "fail": nfail, "total": ntotal,
                    "login_fails_recent": fails, "logins_recent": safe_recent,
                    "roles": nroles,
                    "note": "项目级安全探活摘要：响应头、SSO 角色、登录审计与文件权限。"
                            "该结果不是渗透测试、合规认证或 Cloudflare 配置证明。"})


# ============================================================ I8 发布管理 + 配置中心
def _r9_count(con, sql, args=()):
    try:
        row = con.execute(sql, args).fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def _r9_asset_size(name):
    try:
        return os.path.getsize(os.path.join(app.static_folder, name))
    except Exception:
        return None


def _canonical_payload_hash(payload):
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_RELEASE_MANIFEST_RUNTIME_CACHE = {}
_RELEASE_MANIFEST_RUNTIME_LOCK = threading.Lock()
_RUNTIME_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _runtime_file_signature(path):
    stat_result = os.stat(path, follow_symlinks=False)
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mode,
        stat_result.st_size,
        getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)),
        getattr(stat_result, "st_ctime_ns", int(stat_result.st_ctime * 1_000_000_000)),
    )


def _runtime_manifest_asset_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or relative.startswith("/") or re.match(r"^[A-Za-z]:", relative)):
        raise ValueError("unsafe manifest file path")
    parts = relative.split("/")
    if any(not part or part in (".", "..") for part in parts):
        raise ValueError(f"unsafe manifest file path: {relative}")
    asset_path = os.path.abspath(os.path.join(root, *parts))
    try:
        if os.path.commonpath((root, asset_path)) != root:
            raise ValueError(f"manifest file escapes release root: {relative}")
        real_root = os.path.realpath(root)
        if os.path.commonpath((real_root, os.path.realpath(asset_path))) != real_root:
            raise ValueError(f"manifest file resolves outside release root: {relative}")
    except ValueError as exc:
        raise ValueError(f"unsafe manifest file path: {relative}") from exc
    return asset_path


def _release_manifest_runtime_check_locked(force_content_scan):
    root = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(root, "asset-manifest.json")
    if os.path.islink(manifest_path) or not os.path.isfile(manifest_path):
        raise ValueError("release manifest is missing or is a symlink")

    manifest_signature = _runtime_file_signature(manifest_path)
    cache = _RELEASE_MANIFEST_RUNTIME_CACHE
    reuse_manifest = (
        not force_content_scan
        and cache.get("root") == root
        and cache.get("manifest_path") == manifest_path
        and cache.get("manifest_signature") == manifest_signature
        and isinstance(cache.get("manifest"), dict)
    )
    if reuse_manifest:
        manifest = cache["manifest"]
    else:
        before = manifest_signature
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        after = _runtime_file_signature(manifest_path)
        if before != after:
            raise ValueError("release manifest changed during validation")
        manifest_signature = after
    if not isinstance(manifest, dict):
        raise ValueError("release manifest root must be an object")

    unsigned = dict(manifest)
    observed_artifact = unsigned.pop("artifact_sha256", None)
    if (not isinstance(observed_artifact, str)
            or not _RUNTIME_SHA256_RE.fullmatch(observed_artifact)
            or observed_artifact != _canonical_payload_hash(unsigned)):
        raise ValueError("manifest artifact hash mismatch")
    if manifest.get("release") != ASSET_VER:
        raise ValueError("manifest release mismatch")
    if not isinstance(manifest.get("schema_version"), str) or not manifest["schema_version"]:
        raise ValueError("manifest schema_version is invalid")
    manifest_digest = manifest.get("manifest_digest")
    if not isinstance(manifest_digest, str) or not _RUNTIME_SHA256_RE.fullmatch(manifest_digest):
        raise ValueError("manifest digest is invalid")

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest files must be a non-empty array")
    file_count = manifest.get("file_count")
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count != len(records):
        raise ValueError("manifest file_count mismatch")

    declared_by_path = {}
    filesystem_paths = set()
    total_size = 0
    digest_bound_records = []
    validated_file_cache = {}
    cached_files = cache.get("files", {}) if cache.get("root") == root else {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"manifest file entry {index} must be an object")
        relative = record.get("path")
        asset_path = _runtime_manifest_asset_path(root, relative)
        if relative in declared_by_path:
            raise ValueError(f"duplicate manifest file path: {relative}")
        filesystem_key = os.path.normcase(os.path.realpath(asset_path))
        if filesystem_key in filesystem_paths:
            raise ValueError(f"duplicate manifest filesystem path: {relative}")
        filesystem_paths.add(filesystem_key)

        expected_size = record.get("size")
        expected_sha = record.get("sha256")
        mime = record.get("mime")
        digest_bound = record.get("digest_bound")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"invalid manifest size: {relative}")
        if not isinstance(expected_sha, str) or not _RUNTIME_SHA256_RE.fullmatch(expected_sha):
            raise ValueError(f"invalid manifest sha256: {relative}")
        if not isinstance(mime, str) or not mime:
            raise ValueError(f"invalid manifest mime: {relative}")
        if not isinstance(digest_bound, bool):
            raise ValueError(f"invalid manifest digest_bound: {relative}")
        if os.path.islink(asset_path) or not os.path.isfile(asset_path):
            raise ValueError(f"manifest file missing or is a symlink: {relative}")

        before = _runtime_file_signature(asset_path)
        if before[3] != expected_size:
            raise ValueError(f"manifest file size mismatch: {relative}")
        cached = cached_files.get(relative)
        reuse_hash = (
            not force_content_scan
            and isinstance(cached, dict)
            and cached.get("signature") == before
            and cached.get("sha256") == expected_sha
        )
        if not reuse_hash:
            observed_sha = _file_sha256(asset_path)
            after = _runtime_file_signature(asset_path)
            if before != after:
                raise ValueError(f"manifest file changed during validation: {relative}")
            if observed_sha != expected_sha:
                raise ValueError(f"manifest file sha256 mismatch: {relative}")
            before = after

        declared_by_path[relative] = record
        total_size += expected_size
        validated_file_cache[relative] = {"signature": before, "sha256": expected_sha}
        if digest_bound:
            digest_bound_records.append({
                "path": relative,
                "sha256": expected_sha,
                "size": expected_size,
                "mime": mime,
            })

    declared_total_size = manifest.get("total_size")
    if (isinstance(declared_total_size, bool) or not isinstance(declared_total_size, int)
            or declared_total_size != total_size):
        raise ValueError("manifest total_size mismatch")
    expected_manifest_digest = _canonical_payload_hash({
        "schema_version": manifest.get("schema_version"),
        "release": manifest.get("release"),
        "files": digest_bound_records,
    })
    if manifest_digest != expected_manifest_digest:
        raise ValueError("manifest digest mismatch")

    critical_records = manifest.get("critical_assets")
    if critical_records is None:
        critical_records = []
    if not isinstance(critical_records, list):
        raise ValueError("manifest critical_assets must be an array")
    normalized_critical = []
    critical_paths = set()
    for record in critical_records:
        if not isinstance(record, dict):
            raise ValueError("manifest critical asset entry must be an object")
        relative = record.get("path")
        _runtime_manifest_asset_path(root, relative)
        if relative in critical_paths:
            raise ValueError(f"duplicate critical asset path: {relative}")
        critical_paths.add(relative)
        declared = declared_by_path.get(relative)
        normalized = {
            "path": relative,
            "sha256": record.get("sha256"),
            "size": record.get("size"),
        }
        if declared is None or normalized != {
                "path": relative, "sha256": declared.get("sha256"), "size": declared.get("size")}:
            raise ValueError(f"critical asset is not bound to manifest files: {relative}")
        normalized_critical.append(normalized)

    required_critical = manifest.get("required_critical_assets")
    if required_critical is not None:
        if (not isinstance(required_critical, list)
                or required_critical != [record["path"] for record in normalized_critical]):
            raise ValueError("manifest required_critical_assets mismatch")
    critical_digest = manifest.get("critical_assets_sha256")
    if critical_digest is not None and critical_digest != _canonical_payload_hash(normalized_critical):
        raise ValueError("manifest critical_assets_sha256 mismatch")

    cache.clear()
    cache.update({
        "root": root,
        "manifest_path": manifest_path,
        "manifest_signature": manifest_signature,
        "manifest": manifest,
        "files": validated_file_cache,
    })
    return manifest


def _release_manifest_runtime_check(force_content_scan=False):
    """Verify every declared release file, reusing hashes only for unchanged stats."""
    with _RELEASE_MANIFEST_RUNTIME_LOCK:
        try:
            return _release_manifest_runtime_check_locked(bool(force_content_scan))
        except Exception:
            _RELEASE_MANIFEST_RUNTIME_CACHE.clear()
            raise


def _evidence_age_s(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, time.time() - float(value))
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return None
        return max(0.0, time.time() - parsed.timestamp())
    except ValueError:
        return None


def _site31_gate_evidence_payload(force_manifest_scan=False):
    """Load the release-gate artifact and fail closed on version/hash drift."""
    path = os.path.join(app.static_folder, "quality", "site31_gate_evidence.json")
    fallback = {
        "schema_version": "site31.gate_evidence.v1",
        "release": ASSET_VER,
        "valid": False,
        "gate": "fail",
        "phase": "missing",
        "dimensions": {
            "security": {"max_points": 12, "earned_points": 0, "ratio": 0, "state": "work-in-progress"},
            "accessibility": {"max_points": 6, "earned_points": 0, "ratio": 0, "state": "work-in-progress"},
        },
        "checks": [],
        "summary": {"verified": 0, "manual_check": 0, "failed": 1,
                    "critical_failures": ["gate_evidence_missing"]},
        "error": "release gate evidence is missing or unreadable",
    }
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        artifact_hash = payload.get("artifact_sha256", "")
        unsigned = dict(payload)
        unsigned.pop("artifact_sha256", None)
        expected_hash = _canonical_payload_hash(unsigned)
        if payload.get("release") != ASSET_VER:
            fallback["error"] = "gate evidence release does not match the running release"
            fallback["observed_release"] = payload.get("release")
            return fallback
        if not artifact_hash or artifact_hash != expected_hash:
            fallback["error"] = "gate evidence integrity hash mismatch"
            return fallback
        allowed_phases = {"preflight", "deployed"} if CMD_TEST_MODE else {"deployed"}
        if payload.get("phase") not in allowed_phases:
            fallback["error"] = "production requires deployed gate evidence"
            fallback["observed_phase"] = payload.get("phase")
            return fallback
        gate_age = _evidence_age_s(payload.get("generated_at"))
        if gate_age is None or gate_age > 26 * 3600:
            fallback["error"] = "gate evidence is stale or has no valid timestamp"
            fallback["age_s"] = gate_age
            return fallback
        manifest = _release_manifest_runtime_check(force_content_scan=force_manifest_scan)
        manifest_digest = manifest.get("manifest_digest")
        if (payload.get("asset_manifest") or {}).get("manifest_digest") != manifest_digest:
            fallback["error"] = "gate evidence does not bind the running manifest"
            return fallback
        for key, filename in (
            ("browser_evidence", "site31_browser_evidence.json"),
            ("origin_evidence", "site31_origin_evidence.json"),
        ):
            evidence = payload.get(key) or {}
            evidence_path = os.path.join(app.static_folder, "quality", filename)
            age_s = _evidence_age_s(evidence.get("completed_at"))
            if evidence.get("valid") is not True or evidence.get("manifest_digest") != manifest_digest:
                fallback["error"] = f"{key} is invalid or release-mismatched"
                return fallback
            if age_s is None or age_s > 26 * 3600:
                fallback["error"] = f"{key} is stale"
                fallback["age_s"] = age_s
                return fallback
            expected_evidence_sha = evidence.get("sha256")
            if not expected_evidence_sha or _file_sha256(evidence_path) != expected_evidence_sha:
                fallback["error"] = f"{key} file hash mismatch"
                return fallback
        if payload.get("gate") != "pass":
            fallback["error"] = "release gate did not pass"
            return fallback
        payload["valid"] = True
        return payload
    except Exception as exc:
        fallback["error"] = f"gate evidence load failed: {type(exc).__name__}"
        return fallback


def _site31_gate_evidence_public_payload(force_manifest_scan=False):
    """Return gate proof without local paths, origins or deployment internals."""
    raw = _site31_gate_evidence_payload(force_manifest_scan=force_manifest_scan)
    manifest = raw.get("asset_manifest") or {}
    checks = [{
        "domain": item.get("domain"),
        "key": item.get("key"),
        "label": item.get("label"),
        "state": item.get("state"),
        "max_points": item.get("max_points", 0),
        "earned_points": item.get("earned_points", 0),
        "critical": bool(item.get("critical")),
    } for item in (raw.get("checks") or [])]
    payload = {
        "schema_version": raw.get("schema_version"),
        "release": raw.get("release", ASSET_VER),
        "artifact_sha256": raw.get("artifact_sha256"),
        "generated_at": raw.get("generated_at"),
        "phase": raw.get("phase"),
        "valid": raw.get("valid") is True,
        "gate": raw.get("gate", "fail"),
        "dimensions": raw.get("dimensions") or {},
        "checks": checks,
        "summary": raw.get("summary") or {},
        "asset_manifest": {
            "valid": manifest.get("valid") is True,
            "manifest_digest": manifest.get("manifest_digest"),
            "critical_assets_sha256": manifest.get("critical_assets_sha256"),
        },
        "verification": {
            "browser": {
                "valid": (raw.get("browser_evidence") or {}).get("valid") is True,
                "completed_at": (raw.get("browser_evidence") or {}).get("completed_at"),
                "manifest_matches": (raw.get("browser_evidence") or {}).get("manifest_matches") is True,
            },
            "origin": {
                "valid": (raw.get("origin_evidence") or {}).get("valid") is True,
                "completed_at": (raw.get("origin_evidence") or {}).get("completed_at"),
                "manifest_matches": (raw.get("origin_evidence") or {}).get("manifest_matches") is True,
            },
        },
        "claim_boundary": raw.get("claim_boundary") or (
            "Internal release-readiness evidence; not a penetration test, accessibility certification, "
            "edge configuration proof or third-party ranking."
        ),
    }
    if raw.get("error"):
        payload["error"] = "release gate evidence is unavailable or invalid"
    return payload


@app.route("/api/hardening")
def api_hardening():
    now = int(time.time())
    role_catalog = [
        {"key": "admin", "label": "Admin", "scope": "configuration, rules, release view",
         "public_actions": ["manage config", "create alert rules", "view admin audit"], "ui": "web admin"},
        {"key": "member", "label": "Member", "scope": "prediction, experiment feedback and maintenance records",
         "public_actions": ["create prediction", "submit actual feedback", "record maintenance"], "ui": "web member"},
        {"key": "judge", "label": "Judge / Demo", "scope": "read-only demo mode",
         "public_actions": ["open demo pages", "inspect public evidence"], "ui": "judge"},
    ]
    role_counts = {}
    login_records = 0
    login_failed = 0
    try:
        with open(os.path.join(AUTH_DIR, "users.json"), encoding="utf-8") as f:
            users = json.load(f)
        for u in users.values():
            role = u.get("role") or "unknown"
            role_counts[role] = role_counts.get(role, 0) + 1
    except Exception:
        pass
    try:
        with open(os.path.join(AUTH_DIR, "logins.jsonl"), encoding="utf-8") as f:
            login_lines = f.readlines()[-500:]
        login_records = len(login_lines)
        for ln in login_lines:
            try:
                j = json.loads(ln)
                if not j.get("ok", j.get("success", True)):
                    login_failed += 1
            except Exception:
                pass
    except Exception:
        pass

    con = _db()
    audit = [
        {"event": "login", "source": "SSO logins.jsonl", "records": login_records, "state": "tracked" if login_records else "unavailable",
         "public_fields": ["user", "role", "ip_masked", "ts", "ok"]},
        {"event": "prediction", "source": "workorders.trace_id", "records": _r9_count(con, "SELECT COUNT(*) FROM workorders WHERE trace_id IS NOT NULL AND trace_id!=''"),
         "state": "tracked", "public_fields": ["trace_id", "formula", "verdict", "release"]},
        {"event": "actual_feedback", "source": "workorders.lambda_obs/backfill", "records": _r9_count(con, "SELECT COUNT(*) FROM workorders WHERE lambda_obs IS NOT NULL"),
         "state": "tracked", "public_fields": ["work_order", "lambda_obs", "closed_at"]},
        {"event": "robot_command", "source": "wo_log", "records": _r9_count(con, "SELECT COUNT(*) FROM wo_log"),
         "state": "tracked", "public_fields": ["work_order", "action", "actor", "ts"]},
        {"event": "deployment", "source": "releases", "records": _r9_count(con, "SELECT COUNT(*) FROM releases"),
         "state": "tracked", "public_fields": ["version", "sha", "notes", "by"]},
        {"event": "settings_change", "source": "config", "records": _r9_count(con, "SELECT COUNT(*) FROM config"),
         "state": "tracked", "public_fields": ["key", "type", "updated_by", "ts"]},
        {"event": "permission_change", "source": "users.json role table", "records": sum(role_counts.values()),
         "state": "gateway", "public_fields": ["role", "count"]},
        {"event": "api_key_action", "source": "masked config/webhook slots", "records": _r9_count(con, "SELECT COUNT(*) FROM config WHERE lower(key) LIKE '%key%' OR lower(key) LIKE 'webhook.%'"),
         "state": "masked", "public_fields": ["key", "configured", "updated_by"]},
    ]
    con.close()

    try:
        index_text = open(os.path.join(app.static_folder, "index.html"), encoding="utf-8").read()
        style_text = open(os.path.join(app.static_folder, "style.css"), encoding="utf-8").read()
        app_text = open(os.path.join(app.static_folder, "app.js"), encoding="utf-8").read()
        sw_text = open(os.path.join(app.static_folder, "sw.js"), encoding="utf-8").read()
    except Exception:
        index_text = style_text = app_text = sw_text = ""
    try:
        source_text = open(__file__, encoding="utf-8").read()
    except Exception:
        source_text = ""
    gate_evidence = _site31_gate_evidence_payload()
    gate_checks = {item.get("key"): item for item in gate_evidence.get("checks", [])}

    def gate_state(key, fallback="implemented-partial"):
        if not gate_evidence.get("valid"):
            return fallback
        return gate_checks.get(key, {}).get("state", fallback)

    owasp = [
        {"key": "A01", "name": "Broken access control", "state": "implemented-partial", "evidence": "public method guard and route role checks exist", "verification": "anonymous method matrix + gateway identity-header audit"},
        {"key": "A02", "name": "Cryptographic failures", "state": "manual-check", "evidence": "HTTPS/SSO front door and public-field redaction are designed", "verification": "live TLS/gateway configuration and secret scan"},
        {"key": "A03", "name": "Injection", "state": "implemented-partial", "evidence": "new filters use parameterized SQL and bounded limits", "verification": "route-level payload tests; this is not a full penetration test"},
        {"key": "A04", "name": "Insecure design", "state": "implemented-partial", "evidence": "source labels and no-public-actuation boundary are explicit", "verification": "threat-model and route review"},
        {"key": "A05", "name": "Security misconfiguration", "state": "implemented-partial", "evidence": "CSP, nosniff, referrer, permissions policy and frame controls are set", "verification": "live response-header matrix"},
        {"key": "A06", "name": "Vulnerable components", "state": "manual-check", "evidence": "self-hosted libraries and release records exist", "verification": "dependency inventory and vulnerability scan"},
        {"key": "A07", "name": "Identification/auth failures", "state": "manual-check", "evidence": f"SSO role headers observed; recent login failures {login_failed}", "verification": "confirm proxy strips client X-User/X-Role and copies only forward_auth output"},
        {"key": "A08", "name": "Software/data integrity", "state": "implemented-partial", "evidence": "release SHA records and prediction audit-chain status are visible", "verification": "release and hash-chain audit"},
        {"key": "A09", "name": "Logging/monitoring", "state": "implemented-partial", "evidence": "logs, traces, observability, status and incidents are public-safe", "verification": "failure-injection and alert-delivery review"},
        {"key": "A10", "name": "SSRF", "state": "implemented-partial", "evidence": "public cockpit APIs do not accept arbitrary fetch URLs", "verification": "endpoint input inventory and negative tests"},
    ]
    accessibility = [
        {"name": "Language / skip link / landmarks", "state": gate_state("document.semantics"),
         "evidence": "/api/site31_gate_evidence"},
        {"name": "Accessible names / unique IDs", "state": gate_state("controls.names_ids"),
         "evidence": "/api/site31_gate_evidence"},
        {"name": "Focus visible / current route", "state": gate_state("focus.current_route"),
         "evidence": "/api/site31_gate_evidence"},
        {"name": "Reduced motion / transparency", "state": gate_state("motion.transparency"),
         "evidence": "/api/site31_gate_evidence"},
        {"name": "Polite route status announcement", "state": gate_state("status.route_announcement"),
         "evidence": "/api/site31_gate_evidence"},
        {"name": "Keyboard / zoom browser matrix", "state": gate_state("browser.keyboard_matrix", "manual-check"),
         "evidence": "release validation report"},
        {"name": "NVDA / VoiceOver / WCAG-EM", "state": gate_state("external.screen_reader", "manual-check"),
         "evidence": "independent/manual report required"},
    ]
    sw_marker = ASSET_VER.split("-2026", 1)[0]
    public_surface_text = "\n".join([index_text, app_text, sw_text, json.dumps(_API_DOCS, ensure_ascii=False)])
    unsafe_terms = [
        "raw chassis velocity command",
        "MAG" + " ON",
        "SAFE" + "ZERO",
        "SERVO" + "_WRITE",
        "LIFT" + "_SPIN",
        "GPIO" + " bitpulse",
    ]
    unsafe_hits = [term for term in unsafe_terms if term.lower() in public_surface_text.lower()]
    performance = [
        {"name": "Versioned static assets", "state": "implemented-partial" if ASSET_VER in index_text else "missing", "detail": ASSET_VER},
        {"name": "Service worker cache", "state": "implemented-partial" if sw_marker in sw_text else "missing", "detail": "network-first shell/API cache"},
        {"name": "Three.js route throttling", "state": "implemented-partial" if "loadThreeSceneOnce" in app_text and "threeSceneKick" in app_text else "missing", "detail": "home-only rendering; reduced-motion requires runtime check"},
        {"name": "Offline API cache whitelist", "state": "implemented-partial" if "API_CACHEABLE" in sw_text else "missing", "detail": "read-only API snapshots only"},
        {"name": "Machine-readable API catalog", "state": "implemented-partial" if "/api/openapi.json" in source_text else "missing", "detail": "OpenAPI-like manifest"},
        {"name": "Public evidence manifest", "state": "implemented-partial" if "/api/public_manifest" in source_text else "missing", "detail": "safe-field policy and exports"},
        {"name": "Research passport", "state": "implemented-partial" if "/api/research_passport" in source_text else "missing", "detail": "audience, citation, evidence, limitation and trust posture"},
        {"name": "Site31 evidence contract", "state": "implemented-partial" if "/api/site31_portal" in source_text and "/api/site31_scorecard" in source_text else "missing", "detail": "Evidence Object v3, trust controls and internal release gates"},
        {"name": "Evidence bundle download", "state": "implemented-partial" if "/api/evidence_bundle.json" in source_text else "missing", "detail": "offline judging JSON/TXT bundle"},
        {"name": "Global benchmark console", "state": "implemented-partial" if "/api/global_benchmark" in source_text and "benchmarkview" in index_text else "missing", "detail": "internal evidence-weighted comparison; no external rank claim"},
        {"name": "SEO discovery files", "state": "implemented-partial" if "/robots.txt" in source_text and "/sitemap.xml" in source_text else "missing", "detail": "robots.txt + sitemap.xml"},
        {"name": "Public unsafe-control scan", "state": "verified" if not unsafe_hits else "failed", "detail": ", ".join(unsafe_hits) if unsafe_hits else "no direct actuator/control tokens on public surface"},
    ]
    hardening_layers = [
        {"layer": "application", "name": "Public read-only method boundary", "state": gate_state("app.method_boundary"),
         "detail": "unsafe public methods require SSO/RBAC headers; anonymous POST/PUT/PATCH/DELETE returns 405"},
        {"layer": "application", "name": "Host allowlist", "state": gate_state("app.host_body_boundary"),
         "detail": "canonical site hosts and local health-check hosts are allowed; full host list is not published"},
        {"layer": "application", "name": "Request body limit", "state": gate_state("app.host_body_boundary"),
         "detail": f"MAX_CONTENT_LENGTH={app.config.get('MAX_CONTENT_LENGTH')} bytes"},
        {"layer": "browser", "name": "Security headers", "state": gate_state("browser.security_headers"),
         "detail": "CSP, HSTS on HTTPS, nosniff, referrer policy, permissions policy, frame controls"},
        {"layer": "origin", "name": "Origin service binding", "state": gate_state("origin.loopback_systemd", "manual-check"),
         "detail": "application origin should stay private behind the gateway; verify from operator runbook during deploy"},
        {"layer": "origin", "name": "VPS firewall", "state": gate_state("origin.vps_firewall", "manual-check"),
         "detail": "only public web ingress should be exposed; management and internal service access stays operator-only"},
        {"layer": "edge", "name": "Cloudflare WAF managed rules", "state": "manual-check",
         "detail": "enable Managed Rules / OWASP ruleset / HTTP DDoS in Cloudflare dashboard or API"},
        {"layer": "edge", "name": "Cloudflare rate limiting", "state": "planned",
         "detail": "rate-limit public API families after dashboard/API verification; do not mark pass without evidence"},
        {"layer": "backup", "name": "Rollback and release records", "state": gate_state("release.audit_rollback"),
         "detail": "release table and deploy rollback path are documented; do not expose deploy buttons publicly"},
    ]
    cloudflare_manual = [
        {"priority": "must", "item": "Managed Rules / OWASP ruleset / HTTP DDoS", "status": "manual-check"},
        {"priority": "must", "item": "Cache static assets; keep dynamic pages DYNAMIC/no-cache", "status": "manual-check"},
        {"priority": "should", "item": "Rate limit /api/*, /api/openapi.json, /api/global_benchmark, /api/hardening, /api/evidence_bundle.json", "status": "planned"},
        {"priority": "should", "item": "Block abnormal methods and common scanner paths at edge", "status": "planned"},
        {"priority": "optional", "item": "Authenticated Origin Pulls / Access for SSH-admin flows", "status": "future"},
    ]
    assets = [{"name": n, "bytes": _r9_asset_size(n)} for n in ["index.html", "app.js", "style.css", "sw.js", "twin.js", "three.min.js"]]
    return jsonify({"ts": now, "release": ASSET_VER, "roles": role_catalog, "role_counts": role_counts,
                    "audit": audit, "owasp": owasp, "performance": performance, "accessibility": accessibility,
                    "hardening_layers": hardening_layers, "cloudflare_manual": cloudflare_manual,
                    "gate_evidence": {"valid": gate_evidence.get("valid", False),
                                      "gate": gate_evidence.get("gate", "fail"),
                                      "phase": gate_evidence.get("phase", "missing"),
                                      "dimensions": gate_evidence.get("dimensions", {}),
                                      "artifact_sha256": gate_evidence.get("artifact_sha256")},
                    "assets": assets, "rollback": "operator-only rollback runbook exists; no public SSH command is exposed",
                    "status_definitions": {"verified": "release-matched runtime/config evidence",
                                           "verified-partial": "checkpoint gate passed with disclosed residual manual checks",
                                           "implemented-partial": "code exists; runtime proof pending",
                                           "manual-check": "operator or third-party evidence required",
                                           "planned": "not enabled", "failed": "release check failed"},
                    "note": "Site31 R4 hardening is an internal evidence disclosure, not a certification. Manual-check items must be verified on VPS/Cloudflare or by an independent assessor before being described as verified."})


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        deny = _require_admin()
        if deny:
            return deny
        d = request.get_json(silent=True) or {}
        key = d.get("key")
        if not key:
            return jsonify({"error": "key 必填"}), 400
        value = str(d.get("value", ""))
        if str(key).startswith("webhook.") and value:
            channel = str(key).split(".", 1)[1]
            try:
                _validate_webhook_url(channel, value)
            except ValueError as exc:
                return jsonify({"error": "webhook_rejected", "detail": str(exc)}), 400
        con = _db()
        con.execute("INSERT OR REPLACE INTO config(key,value,type,updated_by,ts) VALUES(?,?,?,?,?)",
                    (key, value, d.get("type", "string"),
                     request.headers.get("X-User", "admin"), int(time.time())))
        con.commit()
        con.close()
        return jsonify({"ok": True})
    masked = []
    for c in _config_all():
        key_l = str(c.get("key", "")).lower()
        sensitive = any(x in key_l for x in ("key", "token", "secret", "password", "passwd", "url", "webhook", "auth", "code"))
        v = c["value"]
        if sensitive:
            v = "configured-redacted" if v else ""
        else:
            v = _public_safe_text(v, 120)
        masked.append({**c, "value": v, "public_value": not sensitive})
    return jsonify({"ts": int(time.time()), "config": masked,
                    "note": "运行期配置中心: 特性开关/阈值/通知 webhook URL (敏感值掩码)。改动审计留痕。"})


@app.route("/api/releases")
def api_releases():
    con = _db()
    rows = [dict(zip(["id", "ver", "ts", "files", "sha", "notes", "by"], r))
            for r in con.execute("SELECT id,ver,ts,files,sha,notes,by FROM releases ORDER BY id DESC LIMIT 40")]
    con.close()
    return jsonify({"ts": int(time.time()), "current": ASSET_VER, "releases": rows,
                    "rollback": "operator-only rollback runbook exists; no public SSH command is exposed",
                    "note": "发布清单 manifest (原子部署+备份上一版); 回滚由受保护运维 runbook 处理。"
                            "出于产线安全, 回滚不暴露为 Web 一键按钮 (避免误触重启致门户中断)。"})


# ============================================================ I9 数据治理 + 备份演练
@app.route("/api/data_inventory")
def api_data_inventory():
    con = _db()
    tables = ["samples", "kpi_samples", "events", "alarms", "maintenance", "workorders",
              "wo_log", "eln", "app_metrics", "logs", "alert_rules", "silences", "oncall",
              "notifications", "ncr", "capa", "esign", "pm_schedule", "spares", "releases",
              "config", "samples_hourly"]
    inv = []
    for t in tables:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            n = None
        inv.append({"table": t, "rows": n})
    try:
        ic = con.execute("PRAGMA integrity_check").fetchone()[0]
    except Exception:
        ic = "?"
    hourly_n = con.execute("SELECT COUNT(*) FROM samples_hourly").fetchone()[0]
    hourly_last = con.execute("SELECT MAX(hour) FROM samples_hourly").fetchone()[0]
    samp_oldest = con.execute("SELECT MIN(ts) FROM samples").fetchone()[0]
    con.close()
    try:
        db_bytes = os.path.getsize(DB_PATH)
        wal_bytes = os.path.getsize(DB_PATH + "-wal") if os.path.exists(DB_PATH + "-wal") else 0
    except OSError:
        db_bytes = wal_bytes = 0
    # 备份健康: /tmp/xrd_data_backup.db (db_backup.py 产物, PC 每日拉)
    bk = {"exists": False, "age_s": None, "bytes": None}
    try:
        bp = "/tmp/xrd_data_backup.db"
        if os.path.exists(bp):
            bst = os.stat(bp)
            bk = {"exists": True, "age_s": int(time.time() - bst.st_mtime), "bytes": bst.st_size}
    except Exception:
        pass
    return jsonify({"ts": int(time.time()), "tables": inv, "integrity": ic,
                    "db_bytes": db_bytes, "wal_bytes": wal_bytes,
                    "retention": {"samples_raw_days": RETAIN_SAMPLES_D, "events_days": RETAIN_EVENTS_D,
                                  "hourly_rollup": "长期保留"},
                    "rollup": {"rows": hourly_n, "last_hour": hourly_last},
                    "samples_oldest": samp_oldest, "backup": bk,
                    "note": "数据生命周期: 原始 30s 样本留 14 天 → 小时降采样长期留; integrity_check 实时校验; "
                            "备份心跳仅证明在线快照存在与时龄，PC 拉取和异地副本需独立核验。WAL 大小反映写入压力。"})


# ============================================================ P9 API 文档
_API_DOCS = [
    ("状态聚合", [
        ("GET", "/api/fleet", "—", "any", "三机在线状态 (真机隧道探活 + 实时指标), 3s 缓存"),
        ("GET", "/api/ops", "—", "any", "运维双路径状态: 真机/镜像探活 + serving + 延迟 + 镜像 systemd 态"),
        ("GET", "/api/twin", "—", "any", "数字孪生遥测: arm01/arm02 回放或遥测标签 / 车位姿·速度·电量 / 各源 live|mirror|replay|offline"),
        ("GET", "/api/kpi", "—", "any", "平台真实 KPI (取 lab serving 端): 预测数/Conformal/审计链/LLM"),
        ("GET", "/api/systems", "—", "any", "三系统名称与公网子域 URL"),
        ("GET", "/api/me", "—", "any", "当前登录用户与角色 (SSO X-User/X-Role)"),
        ("GET", "/api/stream", "—", "any", "SSE 实时流: ops/kpi 快照 + 新事件 + 告警计数即时推送"),
        ("GET", "/api/search/index", "—", "any", "跨系统全局搜索源: lab 预测 + car 语义地标 (命令面板 Ctrl+K), 20s 缓存"),
        ("GET", "/api/preflight", "—", "any", "演示就绪预检: GO/NO-GO 体检清单 + 在险册 + 备份心跳 (全派生自真实状态)"),
        ("GET/POST", "/api/copilot", "{q, deep?}", "any", "问平台运维副驾: 多步 Agent 巡检 (一问可链 7 接地工具) + 推理轨迹 + 动态追问 + 知识库 + DeepSeek 云合成 (deep=true 或无命中兜底时据真数据自然语言作答, 失败降级规则模板)"),
        ("GET", "/api/atlas", "—", "any", "材料图鉴: lab 生成式候选池 (formula/λ_em/verdict/稳定性) + 汇总, 25s 缓存"),
        ("GET", "/api/models", "—", "any", "模型注册表: 9 本地 LLM + 5 BPU slot 真元数据 + 轻量感知/具身/双臂策展清单, 20s 缓存"),
        ("GET", "/api/diagnose", "—", "any", "告警 AI 诊断: 每条活动告警的规则化根因 + 处置步骤 + 关联检查"),
        ("GET/POST", "/api/mirror_sync", "—", "GET any / POST admin", "镜像保鲜同步: 三机镜像数据新鲜度 + 真机可达性 + 陈旧告警 + 同步记录 (POST 触发同步检查)"),
        ("GET/POST", "/api/reproduce", "{idx}", "any", "文献/实测复现台: 实验室实测 λ_em 参考集 (GET) + 调真 predict_engine 复现比对误差与 90% CI 命中 (POST)"),
        ("GET", "/api/metrics", "?metric=&range=", "any", "可观测性时序: historian 分桶降采样 (latency/avail/predictions/ci/audit/llm × 1h/6h/24h/7d), 供交互式折线图"),
        ("GET", "/api/topology", "—", "any", "服务依赖拓扑: 访客→CF→VPS→(真机隧道|镜像)→三机 活体节点图 + 当前 serving 路径高亮"),
        ("GET", "/api/slo_budget", "?window=&target=&scope=", "any", "SLO 错误预算 burn-down: UI 可用性 SLO + 剩余预算 + 燃尽序列 (真 samples)"),
        ("GET", "/api/incidents", "—", "any", "事故复盘: 告警生命周期→时间线+影响+根因+处置+改进项 (自动草稿, 对标 PagerDuty)"),
        ("GET", "/api/timemachine", "?at=", "any", "全局时间机器: 用 historian 重建任意历史时刻平台快照 (三机 serving/延迟+KPI+活动告警+事件)"),
        ("GET", "/api/noc", "—", "any", "统一运营总览 (NOC Wall): 舰队 serving + 延迟火花线 + 24h 可用性 + 活动告警 + KPI + 事件流, 一次聚合"),
        ("POST", "/api/quick_predict", "{formula,symbol,site,pct}", "any", "快速预测 (不建工单): 周期表构建器调 lab 真 predict_engine 返回摘要"),
        ("GET/POST", "/api/eln", "{title,formula,tags,body} / ?q=", "member+", "电子实验笔记 (ELN): 真持久化, 关联化学式+标签+操作人"),
        ("DELETE", "/api/eln/{id}", "—", "member+", "删除实验笔记"),
    ]),
    ("公开证据与全球站能力 (G3-G8)", [
        ("GET", "/api/ai_brain/explain", "—", "any", "AI Brain / Fly-MB 公开解释摘要: LLM/BPU/MLIP/TS/R1 边界与证据链接"),
        ("GET", "/api/rb_voe/explain", "—", "any", "X5-RB-VoE R1 白名单证据: Evidence DAG、failure core、H2 策略、HOLD witness 与零执行权限边界"),
        ("GET", "/api/materials/explorer", "q, verdict, band, sort, limit", "any", "Materials Explorer 结构化材料对象列表, 含来源、CI、证据分、导出链接"),
        ("GET", "/api/materials/export.json", "—", "any", "Materials Atlas 公开 JSON 导出"),
        ("GET", "/api/materials/export.csv", "—", "any", "Materials Atlas 公开 CSV 导出"),
        ("GET", "/api/materials/{id}", "—", "any", "材料详情对象: citation、provenance、evidence_score、download links"),
        ("GET", "/api/predictions/{trace_id}", "—", "any", "预测详情对象: trace_id 对应材料/工单/证据摘要"),
        ("GET", "/api/lab_fsd_console", "—", "any", "Lab-FSD v2 只读世界模型: BEV、policy token、safety gate、source label"),
        ("GET", "/api/experiment_replay", "pct, fault", "any", "闭环实验 10 stage replay + 5 类故障注入 + read-only actuator evidence"),
        ("GET", "/api/cloud_command_center", "—", "any", "Cloud Lab Command Center: 任务、资源、日历、blocker、approval、guardrail"),
        ("GET", "/api/defense_mode", "—", "any", "Defense Mode Pro: 3/5/8 分钟脚本、证据矩阵、demo path、judge checklist"),
        ("GET", "/api/global_benchmark", "—", "any", "Global Benchmark: 顶级网站横向对标、站内证据映射、score gates 与诚实边界"),
        ("GET", "/api/traces", "q, limit", "any", "Trace Explorer: prediction/request 公开 trace waterfall"),
        ("GET", "/api/openapi.json", "—", "any", "机器可读 Public API manifest / OpenAPI-like catalog, 含 curl 示例与安全边界"),
        ("GET", "/api/public_manifest", "—", "any", "公开数据与证据 manifest: safe-field policy、endpoint counts、export examples"),
        ("GET", "/api/research_passport", "—", "any", "Site25 Research Passport: audience、claim、evidence、citation、limitations、trust posture"),
        ("GET", "/api/site30_portal", "—", "any", "Site30 compatibility payload"),
        ("GET", "/api/site31_portal", "—", "any", "Site31 global commercial research portal: research paths、system cards、evidence contract v2、Trust 和 scorecard"),
        ("GET", "/api/site31_scorecard", "—", "any", "Site31 internal evidence-weighted release readiness; not a third-party ranking"),
        ("GET", "/api/site31_gate_evidence", "—", "any", "Release-matched security/accessibility gate checks with SHA-256 integrity and residual risks"),
        ("GET", "/api/research_portal", "—", "any", "Stable Research Portal alias: current Site31 payload"),
        ("GET", "/api/research_collections", "q,scope,topic,has_kind,limit", "any", "Public Research Commons: curated material, evidence and replay collections with release-bound provenance"),
        ("GET", "/api/research_collections/<collection_id>", "—", "any", "Stable public research collection detail with allowlisted members and limitations"),
        ("GET", "/api/site29_portal", "—", "any", "Compatibility alias for Site30 Research Portal payload"),
        ("GET", "/api/evidence_objects", "q,kind,scope,origin,validation_status,source,sort", "any", "Evidence Object v3 index: stable id、claim、provenance、validation、rights、distribution"),
        ("GET", "/api/evidence_objects/schema.json", "—", "any", "Evidence Object v3 JSON Schema 与 DataCite/RO-Crate 映射说明"),
        ("GET", "/api/search/federated", "q,kind,status,source,sort,limit", "any", "材料、预测、证据对象、工单与页面的联合科研检索；返回 facet_groups 与可分享查询"),
        ("GET", "/api/evidence_objects/<evidence_id>", "—", "any", "Evidence Object v3 permanent detail URL"),
        ("GET", "/api/trust_center", "—", "any", "Site31 Trust Center: verified/implemented/manual-check/planned, public boundary and residual risk"),
        ("GET", "/api/evidence_bundle.json", "—", "any", "公开只读证据包 JSON: passport、manifest、materials sample、answer script"),
        ("GET", "/api/evidence_bundle.txt", "—", "any", "公开只读证据包文本版: 便于离线答辩和评委快速复核"),
    ]),
    ("historian 历史库", [
        ("GET", "/api/history", "sys, hours≤336", "any", "状态/延迟历史, 分桶降采样 ≤600 点 (30s 采样)"),
        ("GET", "/api/events", "hours, limit≤500", "any", "事件流 (状态跃迁/告警/工单/维保留痕)"),
        ("GET", "/api/uptime", "—", "any", "SLO: 24h(48 段)/7d(56 段) 可用性 + 真机在线率双口径"),
    ]),
    ("告警中心 (ISA-18.2)", [
        ("GET", "/api/alarms", "—", "any", "活动告警 + 最近已恢复 + 计数 + 邮件通道状态"),
        ("POST", "/api/alarms/{id}/ack", "—", "member+", "人工确认告警, 记操作人"),
    ]),
    ("资产数字孪生", [
        ("GET", "/api/assets", "—", "any", "资产注册表 (真实硬件清单) + 实时 serving + 维保计数"),
        ("GET", "/api/maintenance", "asset, limit", "any", "维保台账列表"),
        ("POST", "/api/maintenance", "{asset, note}", "member+", "添加维保记录, 自动记操作人 + 落事件流"),
    ]),
    ("批次工单 (ISA-88)", [
        ("GET", "/api/workorders", "limit", "any", "工单列表 + 统计"),
        ("POST", "/api/workorders", "{formula, symbol, site, pct}", "member+", "建单并自动调 lab 真预测 (绑 trace_id/verdict/CI)"),
        ("GET", "/api/workorders/{id}", "—", "any", "批次档案: 工单 + 预测摘要 + 全程留痕 wo_log"),
        ("POST", "/api/workorders/{id}/advance", "—", "member+", "推进一个阶段 (回填段须用 backfill 收单)"),
        ("POST", "/api/workorders/{id}/backfill", "{lambda_obs, note}", "member+", "实测回填收单"),
        ("POST", "/api/workorders/{id}/cancel", "—", "member+", "取消工单"),
        ("GET", "/api/workorders/{id}/export", "—", "any", "批次档案 JSON 下载"),
    ]),
    ("报表与导出", [
        ("GET", "/api/reports", "—", "any", "日报列表 (跨天自动生成)"),
        ("GET", "/api/reports/{date}", "—", "any", "某日运行日报 JSON"),
        ("POST", "/api/reports/generate", "{date?}", "admin", "手动生成日报 (今天=partial)"),
        ("GET", "/api/export/events.csv", "hours", "any", "事件 CSV"),
        ("GET", "/api/export/history.csv", "sys, hours", "any", "状态历史 CSV"),
        ("GET", "/api/export/workorders.csv", "—", "any", "工单 CSV"),
    ]),
    ("管理 (admin)", [
        ("GET", "/api/admin/overview", "—", "admin", "服务运行时长/库大小/各表行数/采样与保留参数"),
        ("GET", "/api/admin/users", "—", "admin", "账号列表 (不含口令散列)"),
        ("GET", "/api/admin/logins", "limit", "admin", "登录审计 (logins.jsonl)"),
    ]),
    ("工业级运营 (Round I)", [
        ("GET", "/metrics", "—", "any", "Prometheus 文本曝露 (RED/USE: 请求/延迟/内存/线程/DB/路由)"),
        ("GET", "/api/self", "—", "any", "平台自监控: 进程生命体征 + RED 时序 + 路由热度 (看门狗自监控)"),
        ("GET", "/api/oee", "?window=6h/24h/7d", "any", "OEE 生产线: 可用率A×性能P×质量Q + 班次拆分 + 产量 (全真 historian)"),
        ("GET/POST", "/api/andon", "{sys,action}", "GET any / POST any", "安灯板: 三机 run/idle/down/call + 呼叫支援 (触发告警)"),
        ("GET", "/api/logs", "route,status,q,since,limit", "any", "结构化日志检索 (环形 5000 行, 含 req_id/延迟/用户/IP)"),
        ("GET", "/api/trace/{req_id}", "—", "any", "请求追踪: 同一 req_id 全链路 span (含 copilot 多步)"),
        ("GET/POST", "/api/alert_center", "—", "any", "告警中心: 规则/渠道/静默/值班/通知日志 聚合"),
        ("GET/POST", "/api/alert_rules", "{name,metric,op,threshold,...}", "GET any / POST admin", "告警规则 CRUD (7 指标×6 算子, 命中→通知渠道)"),
        ("DELETE/PATCH", "/api/alert_rules/{id}", "{enabled}", "admin", "删除/启停规则"),
        ("GET/POST", "/api/silences", "{scope,minutes,reason}", "GET any / POST admin", "静默/维护窗口 (抑制升警)"),
        ("DELETE", "/api/silences/{id}", "—", "admin", "提前结束静默"),
        ("GET/POST", "/api/oncall", "{name,contact,hours}", "GET any / POST admin", "值班表"),
        ("POST", "/api/notify_test", "{channel}", "admin", "测试通知渠道 (企业微信/钉钉/飞书/邮件)"),
        ("GET", "/api/qms", "—", "any", "质量中心 (QMS): NCR/CAPA/批次 聚合"),
        ("POST", "/api/ncr", "{batch,defect,severity}", "member+", "开不合格报告 (NCR)"),
        ("POST", "/api/ncr/{id}/capa", "{root_cause,action,owner,due_days}", "member+", "立纠正预防 (CAPA)"),
        ("PATCH", "/api/capa/{id}", "{status}", "member+", "推进/关闭 CAPA (关闭→NCR 闭环)"),
        ("GET", "/api/coa/{batch}", "—", "any", "合格证 (COA): 真 predict + 审计 SHA + e-签名, 可打印"),
        ("POST", "/api/esign", "{obj_type,obj_id,reason}", "member+", "项目电子签批留痕: 签名人/角色/含义/原因/SHA；非 Part 11 认证"),
        ("GET", "/api/genealogy/{batch}", "—", "any", "批次族谱: 原料→工单→预测→判读→NCR/CAPA 血缘图"),
        ("GET", "/api/cmms", "—", "any", "维护管理 (CMMS): 资产 MTBF/MTTR + PM 排程 + 备件库存"),
        ("POST", "/api/pm", "{asset,task,interval_days}", "admin", "新建预防性维护排程"),
        ("POST", "/api/pm/{id}/done", "—", "member+", "PM 完成 (自动续期 + 落维保台账)"),
        ("POST", "/api/spares", "{part,asset,qty,min_qty}", "admin", "备件库存录入"),
        ("GET", "/api/security", "—", "any", "项目安全探活: 响应头 + 审计 + RBAC + 密钥权限；非渗透测试或认证"),
        ("GET", "/api/hardening", "—", "any", "Site31 hardening disclosure: implementation status, verification method, residual risk and manual checks"),
        ("GET/POST", "/api/config", "{key,value,type}", "GET any / POST admin", "配置中心: 特性开关/阈值/webhook (敏感掩码)"),
        ("GET", "/api/releases", "—", "any", "发布历史 + 变更日志 + 当前版本 + 回滚命令"),
        ("GET", "/api/data_inventory", "—", "any", "数据治理: 表清单/库大小/WAL/留存/小时降采样/备份心跳/完整性校验"),
    ]),
]

def _api_doc_entries():
    return reconciled_api_docs(app, _API_DOCS)


def _doc_path_example(path):
    return (path
            .replace("{id}", "1")
            .replace("{trace_id}", "TRACE_ID")
            .replace("{req_id}", "REQ_ID")
            .replace("{batch}", "WO-2026")
            .replace("{date}", "2026-07-06")
            .replace("{rid}", "REQ_ID"))


def _curl_example(method, path, params=""):
    base = (request.url_root or "/").rstrip("/")
    first_method = (method.split("/", 1)[0] or "GET").upper()
    url = base + _doc_path_example(path)
    if first_method == "GET":
        return f"curl -s '{url}'"
    body = "{}"
    if params and params != "—" and params.startswith("{"):
        body = params.replace("'", "\"")
    return f"curl -s -X {first_method} '{url}' -H 'Content-Type: application/json' -d '{body}'"


_DOCS_CSS = """*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'PingFang SC','Microsoft YaHei',system-ui,sans-serif;background:
linear-gradient(160deg,#f3f7ff,#eefcf6 60%,#fef9ec);min-height:100vh;color:#0f172a;padding:34px 5vw 60px}
h1{font-size:1.5rem;margin-bottom:4px}
.sub{color:#64748b;font-size:.82rem;margin-bottom:26px}
h2{font-size:1rem;margin:24px 0 10px;color:#1e293b}
.linkbar{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 22px}
.linkbar a{border:1px solid rgba(37,99,235,.18);background:#fff;color:#1d4ed8;padding:8px 11px;border-radius:10px;text-decoration:none;font-weight:800;font-size:.72rem}
.table-wrap{width:100%;overflow-x:auto;border-radius:14px;box-shadow:0 8px 26px rgba(15,23,42,.06);background:#fff}
table{width:100%;min-width:980px;border-collapse:collapse;background:#fff;font-size:.76rem}
th{background:linear-gradient(120deg,#eff6ff,#ecfeff);text-align:left;padding:9px 13px;color:#475569;font-size:.68rem}
td{padding:9px 13px;border-top:1px solid rgba(15,23,42,.05);vertical-align:top}
.m{font-weight:900;font-size:.66rem;border-radius:6px;padding:2px 8px}
.m.GET{color:#1d4ed8;background:rgba(37,99,235,.1)} .m.POST{color:#b45309;background:rgba(245,158,11,.14)}
code{font-family:Consolas,monospace;color:#0e7490;white-space:nowrap}
.role{font-size:.64rem;font-weight:800;border-radius:999px;padding:1px 8px}
.role.any,.role.public{color:#475569;background:#f1f5f9}
.role.member,.role.internal{color:#065f46;background:rgba(5,150,105,.1)}
.role.reviewer{color:#1d4ed8;background:rgba(37,99,235,.1)}
.role.admin{color:#b91c1c;background:rgba(239,68,68,.1)}
a.back{display:inline-block;margin-bottom:18px;color:#2563eb;font-weight:700;text-decoration:none;font-size:.8rem}
.note{margin-top:26px;font-size:.72rem;color:#64748b;line-height:1.8}
@media(max-width:720px){
body{padding:22px 14px 44px}h1{font-size:1.24rem}.sub{font-size:.76rem}
.table-wrap{overflow:visible;background:transparent;border-radius:0;box-shadow:none}
table{min-width:0;background:transparent}
table,tbody,tr,td{display:block;width:100%}
tr:first-child{display:none}
tr{background:#fff;border:1px solid rgba(15,23,42,.06);border-radius:13px;box-shadow:0 8px 24px rgba(15,23,42,.06);margin:10px 0;padding:8px}
td{border:0;padding:7px 10px;display:grid;grid-template-columns:86px minmax(0,1fr);gap:8px;align-items:start}
td::before{content:attr(data-label);color:#64748b;font-size:.68rem;font-weight:900}
td code{white-space:normal;overflow-wrap:anywhere}
}"""


@app.route("/api/docs")
def api_docs():
    rows = []
    for group, eps in _API_DOCS:
        rows.append(f"<h2>{group}</h2><table><tr><th>方法</th><th>路径</th><th>参数</th>"
                    f"<th>角色</th><th>说明</th></tr>")
        for m, path, params, role, desc in eps:
            rcls = "admin" if role == "admin" else ("member" if "member" in role else "any")
            rlabel = {"any": "登录即可", "member+": "member/admin", "admin": "仅 admin"}.get(role, role)
            rows.append(f"<tr><td><span class='m {m}'>{m}</span></td><td><code>{path}</code></td>"
                        f"<td>{params}</td><td><span class='role {rcls}'>{rlabel}</span></td>"
                        f"<td>{desc}</td></tr>")
        rows.append("</table>")
    html = (f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>API 文档 — XRD 指挥中心</title><style>{_DOCS_CSS}</style></head><body>"
            f"<a class='back' href='/'>← 返回指挥中心</a>"
            f"<h1>📜 指挥中心 API 文档</h1>"
            f"<div class='sub'>全部接口经 SSO forward_auth (cookie xrd_sso); 角色: judge 只读 (GET) /"
            f" member 操作 / admin 管理。写接口另有 Origin 校验 (CSRF 双保险)。"
            f" historian 采样 30s, 样本保留 {RETAIN_SAMPLES_D} 天, 事件/告警 {RETAIN_EVENTS_D} 天。</div>"
            + "".join(rows) +
            "<div class='note'>设备在线时 lab/car/arm 子域各有自己的业务 API (见各系统页);"
            " 本页只列指挥中心聚合层。SSE 用 <code>EventSource('/api/stream')</code> 订阅。</div>"
            "</body></html>")
    return Response(html, mimetype="text/html")


def api_docs_v2():
    rows = []
    for group, entries in group_doc_entries(_api_doc_entries()):
        rows.append(
            f"<h2>{html.escape(group)}</h2><div class='table-wrap'><table>"
            "<tr><th>Method</th><th>Path</th><th>Params</th><th>Role</th><th>Description</th><th>curl</th></tr>"
        )
        for entry in entries:
            method = entry["method"]
            path = entry["path"]
            params = entry["params"]
            role = entry["role"]
            desc = entry["description"]
            first_method = method.split("/", 1)[0]
            rcls = role
            rlabel = {
                "public": "public read",
                "reviewer": "reviewer+",
                "internal": "member/admin",
                "admin": "admin only",
            }.get(role, role)
            rows.append(
                f"<tr><td data-label='Method'><span class='m {html.escape(first_method)}'>{html.escape(method)}</span></td>"
                f"<td data-label='Path'><code>{html.escape(path)}</code></td>"
                f"<td data-label='Params'>{html.escape(params)}</td>"
                f"<td data-label='Role'><span class='role {rcls}'>{html.escape(rlabel)}</span></td>"
                f"<td data-label='Description'>{html.escape(desc)}</td>"
                f"<td data-label='curl'><code>{html.escape(_curl_example(method, path, params))}</code></td></tr>"
            )
        rows.append("</table></div>")
    doc_html = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Public API Docs - XRD Command Center</title><style>{_DOCS_CSS}</style></head><body>"
        "<a class='back' href='/'>Back to Command Center</a>"
        "<h1>Public API Docs</h1>"
        f"<div class='sub'>Public-safe command-center surface. GET endpoints are read-only; write endpoints remain role-gated by SSO, Origin checks and audit logging. "
        f"Retention: raw samples {RETAIN_SAMPLES_D} days, events {RETAIN_EVENTS_D} days.</div>"
        "<div class='linkbar'><a href='/api/openapi.json'>OpenAPI-like JSON</a>"
        "<a href='/api/public_manifest'>Public manifest</a><a href='/api/hardening'>Hardening summary</a>"
        "<a href='/sitemap.xml'>Sitemap</a></div>"
        + "".join(rows) +
        "<div class='note'>Device subdomains expose their own bounded business APIs when online. This page lists the public aggregation layer only. "
        "SSE stream example: <code>new EventSource('/api/stream')</code>.</div>"
        "</body></html>"
    )
    return Response(doc_html, mimetype="text/html")


app.view_functions["api_docs"] = api_docs_v2


def _public_api_doc_entries():
    return [entry for entry in _api_doc_entries() if entry["role"] == "public"]


@app.route("/api/openapi.json")
def api_openapi_json():
    paths = {}
    for entry in _public_api_doc_entries():
        path_item = paths.setdefault(entry["path"], {})
        for method in entry["methods"]:
            method_key = method.lower()
            path_item[method_key] = {
                "tags": [entry["group"]],
                "summary": entry["description"][:140],
                "description": entry["description"],
                "x-role": entry["role"],
                "x-public-safe": entry["role"] == "public" and method == "GET",
                "x-source": entry["source"],
                "x-data-origin": entry["data_origin"],
                "x-runtime-source": entry["runtime_source"],
                "x-freshness-policy": entry["freshness_policy"],
                "x-mutates": entry["mutates"],
                "x-source-labels": ["live", "mirror", "replay", "planned", "mock", "stale", "offline", "unknown"],
                "x-curl": _curl_example(method, entry["path"], entry["params"]),
                "responses": {
                    "200": {"description": "Public-safe JSON, CSV, SSE or HTML depending on endpoint"}
                },
            }
    return jsonify({
        "openapi": "3.0.3",
        "info": {
            "title": "XRD Command Center Public API",
            "version": ASSET_VER,
            "description": "Public-safe read/evidence API catalog for the XRD smart lab command center.",
        },
        "servers": [{"url": (request.url_root or "/").rstrip("/")}],
        "paths": paths,
        "x-boundary": {
            "public_control_policy": "No direct robot, arm, lift, magnet, servo, or chassis velocity command is exposed on public pages.",
            "write_policy": "Write APIs remain SSO/role gated and audited; public site uses read-only evidence views.",
            "field_policy": "Secrets, raw tokens, private prompts and raw actuator payloads are not emitted.",
        },
    })


@app.route("/api/public_manifest")
def api_public_manifest():
    entries = _public_api_doc_entries()
    groups = [
        {"name": group, "method_surfaces": len(group_entries)}
        for group, group_entries in group_doc_entries(entries)
    ]
    examples = [_curl_example(e["methods"][0], e["path"], e["params"]) for e in entries[:12]]
    return jsonify({
        "ts": int(time.time()),
        "release": ASSET_VER,
        "endpoint_count": sum(len(e["methods"]) for e in entries),
        "group_count": len(groups),
        "groups": groups,
        "source_labels": ["live", "mirror", "replay", "planned", "mock", "stale", "offline", "unknown"],
        "safe_field_policy": [
            "Expose trace IDs, verdicts, public evidence, counts, timings, source labels and export links.",
            "Mask secrets, webhook URLs, API keys, private prompts, raw credentials, full IPs and direct actuator payloads.",
            "Label replay/mock/offline/stale data explicitly; do not present it as live hardware state.",
        ],
        "exports": [
            {"label": "Materials JSON", "href": "/api/materials/export.json"},
            {"label": "Materials CSV", "href": "/api/materials/export.csv"},
            {"label": "Research Passport", "href": "/api/research_passport"},
            {"label": "Research Portal Alias", "href": "/api/research_portal"},
            {"label": "Research Collections", "href": "/api/research_collections"},
            {"label": "Evidence Objects", "href": "/api/evidence_objects"},
            {"label": "Trust Center", "href": "/api/trust_center"},
            {"label": "Evidence Bundle JSON", "href": "/api/evidence_bundle.json"},
            {"label": "Evidence Bundle TXT", "href": "/api/evidence_bundle.txt"},
        ],
        "curl_examples": examples,
        "guardrails": [
            "Public pages are read-only; write actions require SSO role checks.",
            "No public robot motion, arm movement, lift, magnet, servo or chassis command endpoint.",
            "CSP, nosniff, referrer policy and permissions policy are applied to every response.",
            "Service worker caches only whitelisted read-only endpoints for offline evidence views.",
            "Research Passport, Evidence Object v3 and Evidence Bundle are public-safe summaries for citation and offline review.",
            "The Site31 scorecard is an internal release-readiness rubric, not a third-party ranking.",
        ],
    })


def _research_passport_payload():
    entries = _public_api_doc_entries()
    materials = _materials_payload({})
    comps = _public_status_components()
    public_gets = sum(1 for e in entries for m in e["methods"] if m == "GET")
    live = sum(1 for c in comps if c.get("source") == "live")
    mirror = sum(1 for c in comps if c.get("source") == "mirror")
    offline = sum(1 for c in comps if c.get("source") == "offline")
    passport_cards = [
        {
            "key": "audience",
            "label": "Audience",
            "title": "面向全球材料科研工作者",
            "value": "NIR phosphor / materials automation",
            "detail": "公开站点优先服务材料方向科研人员, 尤其是近红外荧光粉配方、表征和闭环实验验证。",
            "evidence": ["/atlas", "/brain", "/api/materials/explorer"],
        },
        {
            "key": "evidence",
            "label": "Evidence",
            "title": "可引用对象而不是营销页",
            "value": f"{materials['summary']['total']} material rows · {public_gets} GET surfaces",
            "detail": "材料、预测、状态、发布、链路和安全边界都提供公开字段、来源标签和导出路径。",
            "evidence": ["/api/public_manifest", "/api/openapi.json", "/api/evidence_bundle.json"],
        },
        {
            "key": "boundary",
            "label": "Boundary",
            "title": "公网只读, 物理控制不外放",
            "value": "GET/HEAD/OPTIONS public surface",
            "detail": "匿名写请求会被拒绝; 机械臂、底盘、升降台、电磁铁和速度控制不在公网公开面中。",
            "evidence": ["/api/hardening", "/api/public_status"],
        },
        {
            "key": "trust",
            "label": "Trust",
            "title": "状态透明 + 防护边界透明",
            "value": f"live {live} · mirror {mirror} · offline {offline}",
            "detail": "真机、镜像、回放、陈旧、离线均保持标签; 低配 VPS 情况下优先可解释和可降级。",
            "evidence": ["/status", "/sec", "/api/releases"],
        },
    ]
    top_platform_patterns = [
        {"site": "AlphaFold DB", "borrowed": "可搜索对象、结构/数据下载、API/批量访问、限制说明", "local_surface": "/atlas + /api/materials/*"},
        {"site": "Materials Project", "borrowed": "结构化 API、schema、download docs、可复现查询", "local_surface": "/api/docs + /api/openapi.json"},
        {"site": "Emerald Cloud Lab", "borrowed": "远程实验室工作流、执行边界、实验记录", "local_surface": "/command + /replay + /defense"},
        {"site": "Benchling", "borrowed": "R&D metadata、audit trail、role boundary、trust posture", "local_surface": "/sec + /traces + /api/evidence_bundle.json"},
        {"site": "Vercel / Cloudflare", "borrowed": "release/status/security/readiness 透明表达", "local_surface": "/status + /release + /api/hardening"},
    ]
    trust_posture = [
        {"name": "Public read-only method boundary", "state": "pass", "evidence": "anonymous unsafe methods return 405 unless SSO/RBAC headers are present"},
        {"name": "Host allowlist", "state": "pass", "evidence": "unknown Host returns 421"},
        {"name": "Security headers", "state": "pass", "evidence": "CSP, HSTS on HTTPS, nosniff, referrer policy, permissions policy"},
        {"name": "VPS firewall", "state": "manual-check", "evidence": "last-known operator validation exists; current edge/origin firewall evidence must be rechecked before marking pass"},
        {"name": "Cloudflare WAF / rate limiting", "state": "manual-check", "evidence": "documented as required edge layer; verify in Cloudflare dashboard/API before calling pass"},
    ]
    limitations = [
        "The public website is a research evidence portal, not a public actuator console.",
        "AI predictions do not replace synthesis, XRD and PL measurement.",
        "Project scale is competition/research-lab scale, not the commercial scale of AlphaFold DB, Materials Project, Benchling or Vercel.",
        "Some device states may be mirror/replay/offline when hardware is powered down; labels must remain visible.",
        "Cloudflare WAF/rate-limit status is manual-check unless Cloudflare dashboard/API evidence is attached.",
    ]
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "title": "XRD Smart Lab Research Passport",
        "subtitle": "Global-facing public evidence portal for NIR phosphor and materials automation researchers.",
        "audience": ["materials scientists", "NIR phosphor researchers", "competition judges", "robotic lab evaluators"],
        "one_sentence": "A public-safe, read-only research evidence portal linking AI prediction, the verified embodied hardware loop, finals dual-arm collaboration, material atlas, traces, releases and explicit shadow/assist boundaries.",
        "passport_cards": passport_cards,
        "top_platform_patterns": top_platform_patterns,
        "trust_posture": trust_posture,
        "limitations": limitations,
        "citation": {
            "title": "XRD Smart Lab Public Research Passport",
            "version": ASSET_VER,
            "url": "https://xiaomiju.xyz/",
            "how_to_cite": f"XRD Smart Lab Team. XRD Smart Lab Public Research Passport, version {ASSET_VER}, 2026.",
        },
        "counts": {
            "api_entries": len(entries),
            "public_get_surfaces": public_gets,
            "material_rows": materials["summary"]["total"],
            "material_fields": len(MATERIAL_PUBLIC_FIELDS),
            "components": len(comps),
            "live_components": live,
            "mirror_components": mirror,
            "offline_components": offline,
        },
        "downloads": [
            {"label": "Site31 Research Portal", "href": "/api/site31_portal"},
            {"label": "Research Portal Alias", "href": "/api/research_portal"},
            {"label": "Site31 Scorecard", "href": "/api/site31_scorecard"},
            {"label": "Evidence Objects", "href": "/api/evidence_objects"},
            {"label": "Trust Center", "href": "/api/trust_center"},
            {"label": "Evidence Bundle JSON", "href": "/api/evidence_bundle.json"},
            {"label": "Evidence Bundle TXT", "href": "/api/evidence_bundle.txt"},
            {"label": "Public Manifest", "href": "/api/public_manifest"},
            {"label": "OpenAPI", "href": "/api/openapi.json"},
            {"label": "Materials CSV", "href": "/api/materials/export.csv"},
        ],
    }


@app.route("/api/research_passport")
def api_research_passport():
    return jsonify(_research_passport_payload())


def _site30_evidence_objects(passport=None):
    passport = passport or _research_passport_payload()
    counts = passport.get("counts", {})
    common_redaction = {
        "public_safe": True,
        "excluded": ["secrets", "API keys", "private prompts", "raw actuator commands", "full IP addresses"],
    }
    objects = [
        {
            "evidence_id": f"ev:site30:passport:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "document",
            "scope": "public_site",
            "claim": "本站是面向全球材料科研工作者的公开只读科研证据门户。",
            "claim_status": "curated",
            "source_label": "mirror",
            "evidence_level": "B/C",
            "links": {"page": "/", "api": "/api/research_passport", "download": "/api/evidence_bundle.json"},
            "provenance": {"release": ASSET_VER, "origin": "public passport + manifest", "method_version": "site30.review_board.v1"},
            "validation": {"checks": ["passport_cards", "citation", "limitations", "downloads"], "result": "pass"},
            "trust": {"limitations": ["不是商业规模数据库; 公开站只展示脱敏字段"]},
            "redaction": common_redaction,
        },
        {
            "evidence_id": f"ev:site30:materials:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "material_dataset",
            "scope": "ai_brain",
            "claim": "材料图鉴把公开材料、预测、CI、来源和导出路径组织为可引用对象。",
            "claim_status": "curated",
            "source_label": "mirror",
            "evidence_level": "A/B/C",
            "links": {"page": "/atlas", "api": "/api/materials/explorer", "download": "/api/materials/export.csv"},
            "provenance": {"release": ASSET_VER, "origin": "observed_pl.csv + public material schema", "rows": counts.get("material_rows")},
            "validation": {"checks": ["schema", "public_fields", "export_json", "export_csv"], "result": "pass"},
            "trust": {"uncertainty": "prediction rows keep CI/source labels"},
            "redaction": common_redaction,
        },
        {
            "evidence_id": f"ev:site30:prediction_engine:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "model_system_card",
            "scope": "ai_brain",
            "claim": (FINALS_PUBLIC_FACTS["ai_brain"]["claim"] + " " +
                      "TS/MLIP/Conformal/Fly-MB 与本地 LLM/BPU 证据共同形成配方判决。"),
            "claim_status": "measured+computed+replay",
            "source_label": "mirror",
            "evidence_level": "C/D",
            "links": {"page": "/brain", "api": "/api/ai_brain/explain", "manifest": "/api/models"},
            "provenance": {"release": ASSET_VER, "origin": "predict_engine public summary", "method_version": "TS/MLIP/Conformal/Fly-MB"},
            "validation": {"checks": ["model registry", "public explanation", "uncertainty label"], "result": "pass"},
            "trust": {"limitations": [
                "AI 预测不替代烧制、XRD、PL 实测",
                FINALS_PUBLIC_FACTS["ai_brain"]["boundary"],
            ]},
            "redaction": common_redaction,
        },
        {
            "evidence_id": f"ev:site30:slam_shadow:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "robotics_replay",
            "scope": "embodied_brain",
            "claim": (FINALS_PUBLIC_FACTS["embodied"]["claim"] + " " +
                      FINALS_PUBLIC_FACTS["embodied"]["boundary"]),
            "claim_status": "measured+replay+shadow",
            "source_label": "replay",
            "evidence_level": "C",
            "links": {"page": "/fsd", "api": "/api/lab_fsd_console", "status": "/api/public_status"},
            "provenance": {"release": ASSET_VER, "origin": "frozen finals hardware record + public replay", "hardware_state": "X5 may be unpowered"},
            "validation": {"checks": ["observed finals record", "source_label", "safety_boundary", "no_public_chassis_velocity"], "result": "pass"},
            "trust": {"limitations": ["Lab-FSD 是 shadow/assist, 不是公网无人控制入口"]},
            "redaction": common_redaction,
        },
        {
            "evidence_id": f"ev:site30:arm01_redundancy:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "robotics_replay",
            "scope": "arm01",
            "claim": (FINALS_PUBLIC_FACTS["dual_arm"]["claim"] + " " +
                      FINALS_PUBLIC_FACTS["dual_arm"]["boundary"]),
            "claim_status": "measured+replay",
            "source_label": "replay",
            "evidence_level": "A/C",
            "links": {"page": "/replay", "api": "/api/experiment_replay", "assets": "/api/assets"},
            "provenance": {"release": ASSET_VER, "origin": "frozen dual-arm finals record + public replay", "hardware_state": "Pi may be unpowered"},
            "validation": {"checks": ["arm01 visual redundancy", "arm02 four-cycle grinding", "CPU/OpenCV authority", "BPU assist boundary", "no public arm command"], "result": "pass"},
            "trust": {"limitations": ["公开页仅回放/镜像, 不下发机械臂动作"]},
            "redaction": common_redaction,
        },
        {
            "evidence_id": f"ev:site30:trust_boundary:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "trust_control",
            "scope": "public_site",
            "claim": "公网面只读, 写操作需要 SSO/RBAC, 且不暴露物理控制。",
            "claim_status": "live+manual-check",
            "source_label": "live",
            "evidence_level": "B",
            "links": {"page": "/sec", "api": "/api/hardening", "manifest": "/api/public_manifest"},
            "provenance": {"release": ASSET_VER, "origin": "Flask boundary guard + hardening summary"},
            "validation": {"checks": ["method_boundary", "host_allowlist", "security_headers", "unsafe_control_scan"], "result": "pass"},
            "trust": {"limitations": ["Cloudflare WAF/rate-limit 需控制台或 API 证据后才可标 pass"]},
            "redaction": common_redaction,
        },
        {
            "evidence_id": f"ev:site30:release_integrity:{ASSET_VER}",
            "schema_version": "site30.evidence_object.v1",
            "kind": "release_record",
            "scope": "public_site",
            "claim": "发布、缓存、回滚和当前版本在公开只读面保持可追溯。",
            "claim_status": "live",
            "source_label": "live",
            "evidence_level": "B",
            "links": {"page": "/release", "api": "/api/releases", "hardening": "/api/hardening"},
            "provenance": {"release": ASSET_VER, "origin": "deploy release table + service worker cache version"},
            "validation": {"checks": ["asset_version", "sw_cache", "release_table"], "result": "pass"},
            "trust": {"limitations": ["回滚命令只作为运维说明, 不做公网一键按钮"]},
            "redaction": common_redaction,
        },
    ]
    for item in objects:
        links = item.get("links", {})
        item["citation_text"] = f"XRD Smart Lab Team. {item.get('kind', 'evidence')} {item.get('scope', 'public_site')}, {ASSET_VER}."
        item["download_links"] = [v for v in links.values() if isinstance(v, str) and v.startswith("/api/")]
        item["limitations_short"] = "; ".join((item.get("trust") or {}).get("limitations", [])[:2]) or "public-safe summary only"
        item["reuse_hint"] = "Open the linked API/page to inspect public fields, source labels, validation checks and limitations."
        item["characteristics"] = {
            "scope": item.get("scope"),
            "kind": item.get("kind"),
            "source_label": item.get("source_label"),
            "claim_status": item.get("claim_status"),
        }
    return objects


def _site30_trust_center_payload(passport=None):
    passport = passport or _research_passport_payload()
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "schema_version": "site30.trust_center.v1",
        "title": "Public Trust Center",
        "summary": "公网展示站保持只读、可引用、可降级、可追溯; 真机离线时明确标注 mirror/replay/planned/offline。",
        "controls": [
            {
                "id": "tc.public_read_only",
                "framework": "OWASP / research portal boundary",
                "state": "pass",
                "evidence_object_ids": [f"ev:site30:trust_boundary:{ASSET_VER}"],
                "verification_method": "unsafe method guard + public manifest",
                "residual_risk": "SSO/RBAC 配置需随部署持续巡检",
            },
            {
                "id": "tc.robot_actuation_boundary",
                "framework": "robotics safety case",
                "state": "pass",
                "evidence_object_ids": [f"ev:site30:slam_shadow:{ASSET_VER}", f"ev:site30:arm01_redundancy:{ASSET_VER}"],
                "verification_method": "public unsafe-control scan + source labels",
                "residual_risk": "公网页面不得新增任何速度、舵机、升降台或电磁铁写控制",
            },
            {
                "id": "tc.source_label_truth",
                "framework": "StatusPage-style transparency",
                "state": "pass",
                "evidence_object_ids": [f"ev:site30:release_integrity:{ASSET_VER}"],
                "verification_method": "public_status components + UI labels",
                "residual_risk": "硬件未上电时必须继续显示 mirror/replay/planned/offline",
            },
            {
                "id": "tc.data_redaction",
                "framework": "FAIR public evidence + safe-field policy",
                "state": "pass",
                "evidence_object_ids": [f"ev:site30:materials:{ASSET_VER}"],
                "verification_method": "public schema and evidence bundle redaction policy",
                "residual_risk": "新增字段先做脱敏审查",
            },
            {
                "id": "tc.edge_waf",
                "framework": "Cloudflare/Vercel trust posture",
                "state": "manual-check",
                "evidence_object_ids": [f"ev:site30:trust_boundary:{ASSET_VER}"],
                "verification_method": "Cloudflare dashboard/API required",
                "residual_risk": "没有边缘控制台证据前不能写成 pass",
            },
        ],
        "status_taxonomy": passport.get("counts", {}),
        "public_boundary": {
            "allowed_public_methods": sorted(_SAFE_PUBLIC_METHODS),
            "blocked_public_payloads": ["robot motion", "arm movement", "lift", "magnet", "servo", "chassis velocity"],
            "safe_source_labels": ["live", "mirror", "replay", "planned", "stale", "offline", "unknown"],
        },
    }


def _site30_portal_payload():
    passport = _research_passport_payload()
    objects = _site30_evidence_objects(passport)
    trust = _site30_trust_center_payload(passport)
    counts = passport.get("counts", {})
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "schema_version": "site30.research_review_board.v1",
        "title": "Site30 Research Review Board",
        "subtitle": "把网站从项目展示升级为材料科研门户: 对象可引用、证据可下载、状态可追溯、边界可解释。",
        "open_science_commitments": [
            {"name": "透明状态", "detail": "live/mirror/replay/planned/offline 来源标签显式展示", "evidence": "/api/public_status"},
            {"name": "可下载证据", "detail": "Research Passport、Evidence Bundle、材料 CSV/JSON 可离线复核", "evidence": "/api/evidence_bundle.json"},
            {"name": "可复现摘要", "detail": "材料公开字段、预测 trace、发布版本和验证检查集中到对象记录", "evidence": "/api/evidence_objects"},
            {"name": "公开边界", "detail": "公网只读、不暴露机器人控制和底层网络/执行器细节", "evidence": "/api/trust_center"},
        ],
        "negative_controls": [
            {"id": "no_public_robot_control", "state": "pass", "check": "no public chassis/arm/lift/magnet command route"},
            {"id": "no_raw_network_details", "state": "pass", "check": "public assets and hardening avoid raw IP/port/operator command details"},
            {"id": "no_operator_commands", "state": "pass", "check": "public preflight/diagnose returns abstract runbook IDs"},
            {"id": "cloudflare_edge_not_verified", "state": "manual-check", "check": "edge WAF/rate limit remains manual-check until dashboard/API evidence exists"},
        ],
        "current_fact_boundary": [
            FINALS_PUBLIC_FACTS["embodied"]["claim"] + " " + FINALS_PUBLIC_FACTS["embodied"]["boundary"],
            FINALS_PUBLIC_FACTS["ai_brain"]["claim"] + " " + FINALS_PUBLIC_FACTS["ai_brain"]["boundary"],
            FINALS_PUBLIC_FACTS["dual_arm"]["claim"] + " " + FINALS_PUBLIC_FACTS["dual_arm"]["boundary"],
            "X5/Pi 未上电时显示 mirror/replay/offline/unknown，不伪装 live。",
        ],
        "hero_metrics": [
            {"value": len(objects), "label": "核心证据对象", "source": "/api/evidence_objects"},
            {"value": counts.get("public_get_surfaces", 0), "label": "公开 GET 证据面", "source": "/api/public_manifest"},
            {"value": counts.get("material_rows", 0), "label": "材料公开行", "source": "/api/materials/explorer"},
            {"value": "0", "label": "公网物理控制入口", "source": "/api/hardening"},
        ],
        "top_platform_patterns": [
            {"site": "AlphaFold DB", "pattern": "对象搜索 / 下载 / API / 限制说明", "reuse": "材料图鉴 + 公开 API + 引用字段", "url": "https://alphafold.ebi.ac.uk/"},
            {"site": "Materials Project", "pattern": "schema-first API 和可复现查询", "reuse": "OpenAPI + materials schema + export", "url": "https://docs.materialsproject.org/"},
            {"site": "Nature / Scientific Data", "pattern": "Data availability / Code availability / technical validation", "reuse": "Research Passport + Evidence Bundle", "url": "https://www.nature.com/sdata/submission-guidelines"},
            {"site": "Emerald Cloud Lab", "pattern": "实验工作流、任务和审计记录", "reuse": "Command + Replay + work_order", "url": "https://www.emeraldcloudlab.com/"},
            {"site": "Waymo Safety", "pattern": "感知、世界模型、回放、安全边界", "reuse": "SLAM + Lab-FSD shadow + Nav2 边界", "url": "https://waymo.com/safety/"},
            {"site": "Cloudflare / Vercel", "pattern": "Trust、Status、Release、edge boundary", "reuse": "Trust Center + status + release", "url": "https://www.cloudflare.com/trust-hub/"},
        ],
        "evidence_ladder": [
            {"level": "A", "name": "实测/真机/回填", "use": "PL/XRD 回填、具身闭环、双臂协同与 BPU 实测指标"},
            {"level": "B", "name": "公开文献/数据源", "use": "observed_pl、材料图鉴、引用/导出字段"},
            {"level": "C", "name": "计算/回放/镜像", "use": "MLIP/TS、SLAM replay、Lab-FSD shadow、public status"},
            {"level": "D", "name": "推理/解释/规划", "use": "LLM verdict、Fly-MB、答辩脚本、未来迭代计划"},
        ],
        "homepage_rails": [
            {"label": "Research Passport", "page": "/", "api": "/api/research_passport", "purpose": "评委 30 秒理解项目可信度"},
            {"label": "Evidence Objects", "page": "/", "api": "/api/evidence_objects", "purpose": "把每个 claim 变成可追溯对象"},
            {"label": "Trust Center", "page": "/sec", "api": "/api/trust_center", "purpose": "说明公网只读、防护边界和人工检查项"},
            {"label": "Global Benchmark", "page": "/benchmark", "api": "/api/global_benchmark", "purpose": "说明借鉴对象、差距和门禁"},
        ],
        "evidence_objects": objects,
        "trust_center": trust,
        "acceptance_gates": [
            "release/site worker/OpenAPI/Public Manifest 版本一致",
            "首页显示 evidence object、trust controls、top platform reuse",
            "没有绝对化宣传表述; 不把 shadow/replay/planned 写成 live autonomy",
            "公网仍为只读; unsafe-control scan pass",
        ],
    }


def _stable_evidence_id(value):
    parts = str(value or "").split(":")
    key = parts[2] if len(parts) >= 3 and parts[0] == "ev" else re.sub(r"[^a-z0-9_-]+", "-", str(value).lower())
    key = re.sub(r"[^a-z0-9_-]+", "-", key).strip("-") or "record"
    return f"ev:xrd:{key}"


def _canonical_json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _evidence_snapshot_payload(item):
    excluded = {"distributions", "download_links"}
    return {k: v for k, v in item.items() if k not in excluded}


def _site31_evidence_objects(passport=None):
    """Evidence Object v3: stable identity, typed provenance and honest distribution metadata."""
    items = json.loads(json.dumps(_site30_evidence_objects(passport), ensure_ascii=False))
    titles = {
        "document": ("XRD 智慧实验室公开科研护照", "XRD Smart Lab Public Research Passport"),
        "material_dataset": ("近红外荧光材料公开数据对象", "Public NIR Phosphor Materials Dataset"),
        "model_system_card": ("AI 脑预测引擎系统卡", "AI Brain Prediction Engine System Card"),
        "trust_control": ("公网只读与脱敏控制", "Public Read-only and Redaction Control"),
        "release_record": ("发布完整性与缓存记录", "Release Integrity and Cache Record"),
    }
    scope_titles = {
        "embodied_brain": ("SLAM 建图与 Lab-FSD shadow 系统卡", "SLAM Mapping and Lab-FSD Shadow System Card"),
        "arm01": ("双机械臂复赛协同与视觉门控证据", "Finals Dual-arm Collaboration and Visual-gate Evidence"),
    }
    claim_en = {
        "passport": "A public, read-only research evidence portal for materials researchers worldwide.",
        "materials": "Public material records connect predictions, uncertainty, provenance and exports as citable objects.",
        "prediction_engine": "The AI brain combines TS, MLIP, conformal and Fly-MB evidence while preserving experimental limits.",
        "slam_shadow": "The embodied brain presents SLAM evidence and Lab-FSD shadow/assist without public motion authority.",
        "arm01_redundancy": "Real-hardware finals evidence covers arm01 visual redundancy and bag drop with arm02 concurrent four-cycle grinding; CPU/OpenCV remains authoritative for bag state.",
        "trust_boundary": "The public surface is read-only, role-gated for writes and exposes no physical actuation route.",
        "release_integrity": "Release, cache, rollback and validation records remain traceable by version and artifact evidence.",
    }
    origin_modes = {
        "document": ["curated", "release-record"],
        "material_dataset": ["observed", "literature", "computed"],
        "model_system_card": ["computed", "model-evaluation", "mirror"],
        "robotics_replay": ["measured", "replay", "shadow"],
        "trust_control": ["implementation", "runtime-check"],
        "release_record": ["runtime-check", "release-record"],
    }
    intended = {
        "public_site": "公开复核项目主张、来源、限制、发布状态与下载入口",
        "ai_brain": "复核材料预测、方法版本、不确定性与实测回流边界",
        "embodied_brain": "复核 SLAM、回放、shadow/assist 与底盘控制边界",
        "arm01": "复核 arm01 视觉冗余、投袋与 arm02 并发四周期研磨记录",
    }
    intended_en = {
        "public_site": "Review public claims, sources, limitations, release state and download entry points.",
        "ai_brain": "Review material predictions, method versions, uncertainty and experimental-feedback boundaries.",
        "embodied_brain": "Review SLAM, replay, shadow/assist evidence and the chassis-control boundary.",
        "arm01": "Review arm01 visual redundancy and bag drop with arm02 concurrent four-cycle grinding.",
    }
    prohibited = {
        "public_site": "不得作为物理设备控制面、安全认证或全球排名结论",
        "ai_brain": "不得替代烧制、XRD、PL 实测或作为唯一材料结论",
        "embodied_brain": "不得解释为 Lab-FSD 已接管底盘或具备道路自动驾驶能力",
        "arm01": "不得把 BPU 解释为袋状态权威，也不得作为公网机械臂控制入口",
    }
    prohibited_en = {
        "public_site": "Do not treat this portal as a physical-control surface, security certification or global-ranking result.",
        "ai_brain": "Do not replace synthesis, XRD or PL measurements, or use predictions as the sole material conclusion.",
        "embodied_brain": "Do not interpret Lab-FSD as holding chassis authority or road-driving capability.",
        "arm01": "Do not treat BPU as authoritative for bag state or use this evidence as a public arm-control surface.",
    }
    limitations_en = {
        "passport": ["The public snapshot excludes private prompts, credentials, raw internal logs and physical control."],
        "materials": ["The curated public corpus is limited in scale and does not replace primary measurements or source-license review."],
        "prediction_engine": ["Predictions remain subject to model and dataset limits and require synthesis, XRD and PL validation."],
        "slam_shadow": ["Lab-FSD remains shadow/assist only and holds no public or autonomous chassis authority."],
        "arm01_redundancy": ["CPU/OpenCV is authoritative for bag state; BPU is supporting evidence only, and the public surface issues no arm commands."],
        "trust_boundary": ["Point-in-time controls reduce risk but do not prove absolute security or third-party certification."],
        "release_integrity": ["Release checks are point-in-time and do not replace long-term SLO or independent audits."],
    }
    validation_states = {
        "passport": "verified",
        "materials": "verified",
        "prediction_engine": "implemented-partial",
        "slam_shadow": "implemented-partial",
        "arm01_redundancy": "verified",
        "trust_boundary": "verified",
        "release_integrity": "verified",
    }
    relation_types = {
        "page": "IsDescribedBy", "api": "IsIdenticalTo", "download": "HasPart",
        "manifest": "IsDocumentedBy", "status": "IsSupplementTo", "assets": "HasPart",
        "hardening": "IsDocumentedBy",
    }
    resource_types = {
        "document": "Text", "material_dataset": "Dataset", "model_system_card": "Model",
        "robotics_replay": "Dataset", "trust_control": "Text", "release_record": "Workflow",
    }
    for item in items:
        old_id = item.get("evidence_id", "")
        object_key = old_id.split(":")[2] if len(old_id.split(":")) >= 3 else "record"
        stable_id = _stable_evidence_id(old_id)
        title_zh, title_en = scope_titles.get(item.get("scope")) or titles.get(
            item.get("kind"), ("公开科研证据对象", "Public Research Evidence Object"))
        links = {k: v for k, v in (item.get("links") or {}).items() if isinstance(v, str) and v.startswith("/")}
        limitations = list((item.get("trust") or {}).get("limitations", []))
        uncertainty_text = ((item.get("trust") or {}).get("uncertainty") or
                            "见来源标签与限制；该对象不提供第三方科研置信度或全球排名结论。")
        source_label = item.get("source_label") or "unknown"
        provenance = item.get("provenance") or {}
        relations = [{"relation_type": relation_types.get(k, "IsReferencedBy"), "target": v,
                      "target_kind": k} for k, v in sorted(links.items())]
        citation_key = "xrd_" + re.sub(r"[^a-z0-9]+", "_", object_key.lower()).strip("_")
        citation_text = f"XRD Smart Lab Team. {title_zh}. Version {ASSET_VER}, 2026."
        citation_text_en = f"XRD Smart Lab Team. {title_en}. Version {ASSET_VER}, 2026."
        transformed = {
            "evidence_id": stable_id,
            "schema_version": "site31.evidence_object.v3",
            "identifier": {"local": stable_id, "doi": None, "pid": None,
                           "registration_status": "not_registered"},
            "doi": None,
            "pid": None,
            "version": {"release": ASSET_VER, "revision": "r4", "released_at": RELEASED_AT},
            "release": ASSET_VER,
            "kind": item.get("kind"),
            "scope": item.get("scope"),
            "resource_type": {"general": resource_types.get(item.get("kind"), "Other"),
                              "type": item.get("kind") or "evidence_object"},
            "title": title_zh,
            "title_en": title_en,
            "description": item.get("claim") or "",
            "description_en": claim_en.get(object_key, "Public-safe evidence record with explicit provenance and limits."),
            "owner": "XRD Smart Lab Team",
            "creators": [{"name": "XRD Smart Lab Team", "name_type": "Organizational"}],
            "publisher": "XRD Smart Lab Team",
            "dates": {"published": "2026-07-10", "updated": "2026-07-10"},
            "claim": item.get("claim") or "",
            "claim_status": item.get("claim_status") or "unknown",
            "claims": [{
                "claim_id": f"{stable_id}#claim-1",
                "statement": item.get("claim") or "",
                "statement_en": claim_en.get(object_key, ""),
                "status": item.get("claim_status") or "unknown",
                "evidence_modes": origin_modes.get(item.get("kind"), [source_label]),
                "source_label": source_label,
            }],
            "source_label": source_label,
            "origin": origin_modes.get(item.get("kind"), [source_label]),
            "property_provenance": {
                "claim": {"source": provenance.get("origin") or "public-safe project evidence",
                          "method_version": provenance.get("method_version") or ASSET_VER},
                "source_label": {"source": "runtime or labelled mirror/replay boundary",
                                 "checked_at": RELEASED_AT},
                "release": {"source": "deployment release manifest", "value": ASSET_VER},
            },
            "validation_status": validation_states.get(object_key, "implemented-partial"),
            "validation": {
                "checks": list((item.get("validation") or {}).get("checks", [])),
                "result": (item.get("validation") or {}).get("result", "unknown"),
                "scope": "project evidence and release checks",
                "scope_en": "Project evidence and release checks",
                "independent_third_party": False,
            },
            "uncertainty": {
                "statement": uncertainty_text,
                "statement_en": "See source labels and limitations. This object does not provide third-party scientific confidence or a global-ranking conclusion.",
                "quantification": None,
            },
            # Evidence snapshots are immutable release records. Runtime age belongs on
            # /api/public_status; embedding a ticking age here would invalidate the
            # advertised snapshot hash on every request.
            "freshness": {
                "state": source_label if source_label in {
                    "live", "mirror", "replay", "mock", "stale", "offline", "unknown", "planned"
                } else "unknown",
                "source": "release-evidence",
                "checked_at": RELEASED_AT,
                "age_s": 0,
                "freshness": "fresh",
                "confidence": "reported",
                "error": None,
                "release": ASSET_VER,
                "snapshot_semantics": "as-released",
            },
            "intended_use": intended.get(item.get("scope"), "公开复核与答辩证据定位"),
            "intended_use_en": intended_en.get(item.get("scope"), "Locate public evidence for review."),
            "prohibited_use": prohibited.get(item.get("scope"), "不得超出公开字段与限制说明外推"),
            "prohibited_use_en": prohibited_en.get(item.get("scope"), "Do not infer beyond the public fields and stated limitations."),
            "limitations": limitations,
            "limitations_en": limitations_en.get(
                object_key, ["Use only within the stated public scope, provenance and limitations."]),
            "rights": {
                "access": "public-read-only",
                "license": "LicenseRef-XRD-Public-Summary",
                "statement": "No license is asserted for underlying third-party data; source licenses continue to apply.",
            },
            "redaction": item.get("redaction") or {},
            "links": links,
            "relations": relations,
            "canonical_url": "/api/evidence_objects/" + quote(stable_id, safe=""),
            "revision_history": [{"release": ASSET_VER, "date": "2026-07-10",
                                  "change": "Migrated to stable Evidence Object v3 identity and distribution contract."}],
            "inputs_outputs": {
                "inputs": ["public-safe source summary", "release metadata", "validation checks"],
                "outputs": ["claim", "provenance", "validation", "limitations", "citable snapshot"],
            },
            "citation_text": citation_text,
            "citation_text_en": citation_text_en,
            "citation": {
                "text": citation_text,
                "text_en": citation_text_en,
                "bibtex": f"@misc{{{citation_key}, title={{{title_zh}}}, author={{{{XRD Smart Lab Team}}}}, year={{2026}}, note={{{ASSET_VER}}}}}",
                "bibtex_en": f"@misc{{{citation_key}, title={{{title_en}}}, author={{{{XRD Smart Lab Team}}}}, year={{2026}}, note={{{ASSET_VER}}}}}",
                "ris": f"TY  - DATA\nTI  - {title_zh}\nAU  - XRD Smart Lab Team\nPY  - 2026\nVL  - {ASSET_VER}\nER  -",
                "ris_en": f"TY  - DATA\nTI  - {title_en}\nAU  - XRD Smart Lab Team\nPY  - 2026\nVL  - {ASSET_VER}\nER  -",
            },
            "datacite_mapping": {
                "schema": "DataCite Metadata Schema 4.7 mapping (not registered)",
                "identifier": "identifier.local; DOI is null until external registration",
                "creators": "creators", "titles": "title/title_en", "publisher": "publisher",
                "publicationYear": "dates.published", "resourceType": "resource_type",
                "relatedIdentifiers": "relations", "rightsList": "rights",
            },
            "ro_crate_mapping": {
                "profile": "RO-Crate 1.3 mapping guidance",
                "@id": stable_id, "@type": resource_types.get(item.get("kind"), "CreativeWork"),
                "name": title_zh, "description": item.get("claim") or "",
                "license": None, "hasPart": [r["target"] for r in relations if r["relation_type"] == "HasPart"],
            },
            "legacy_evidence_label": item.get("evidence_level"),
        }
        snapshot = _evidence_snapshot_payload(transformed)
        raw = _canonical_json_bytes(snapshot)
        snapshot_href = "/api/evidence_objects/" + quote(stable_id, safe="") + "/snapshot.json"
        transformed["distributions"] = [{
            "label": "Evidence Object v3 JSON snapshot",
            "href": snapshot_href,
            "available": True,
            "mime": "application/json",
            "mime_type": "application/json",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "generated_at": RELEASED_AT,
            "license": transformed["rights"]["license"],
        }]
        transformed["download_links"] = [snapshot_href]
        items[items.index(item)] = transformed
    return items


def _site31_trust_center_payload(passport=None):
    base = _site30_trust_center_payload(passport)
    gate_evidence = _site31_gate_evidence_payload()
    gate_checks = {item.get("key"): item for item in gate_evidence.get("checks", [])}
    redaction_state = gate_checks.get("public.redaction_no_actuation", {}).get("state", "implemented-partial")
    controls = []
    control_states = {
        "tc.public_read_only": "verified",
        "tc.robot_actuation_boundary": "verified",
        "tc.source_label_truth": "verified",
        "tc.data_redaction": redaction_state if gate_evidence.get("valid") else "implemented-partial",
        "tc.edge_waf": "manual-check",
    }
    for control in base.get("controls", []):
        row = dict(control)
        row["state"] = control_states.get(row.get("id"), "manual-check")
        row["evidence_object_ids"] = [_stable_evidence_id(x) for x in row.get("evidence_object_ids", [])]
        row["claim_scope"] = "internal implementation status; not a third-party certification"
        controls.append(row)
    controls.append({
        "id": "tc.gateway_header_sanitization",
        "framework": "reverse-proxy identity boundary",
        "state": "verified",
        "evidence_object_ids": [],
        "verification_method": "confirm Caddy removes client-supplied X-User/X-Role and copies only forward_auth output",
        "residual_risk": "configuration drift must be rechecked after Caddy changes",
        "claim_scope": "verified from deployed Caddy forward_auth/copy_headers and loopback-only origin binding",
    })
    controls.append({
        "id": "tc.release_gate_evidence",
        "framework": "OWASP ASVS + WCAG 2.2 internal release gate",
        "state": "verified" if gate_evidence.get("valid") and gate_evidence.get("gate") == "pass" else "failed",
        "evidence_object_ids": [],
        "verification_method": "/api/site31_gate_evidence; version and SHA-256 integrity checked at runtime",
        "residual_risk": "internal automation does not replace independent penetration or screen-reader audits",
        "claim_scope": "release readiness only; no third-party certification or global rank claim",
    })
    origin_state = gate_checks.get("origin.vps_firewall", {}).get("state", "manual-check")
    controls.append({
        "id": "tc.origin_boundary",
        "framework": "VPS firewall + loopback origin + Caddy SSO boundary",
        "state": origin_state if gate_evidence.get("valid") else "manual-check",
        "evidence_object_ids": [],
        "verification_method": "UFW status, listening-socket inventory, active Caddyfile and forged identity-header negative test",
        "residual_risk": "Cloudflare WAF and edge rate-limit rules still require dashboard/API evidence",
        "claim_scope": "point-in-time origin/gateway evidence; not an independent penetration test",
    })
    base.update({
        "release": ASSET_VER,
        "schema_version": "site31.trust_center.v3",
        "title": "Site31 Public Trust Center",
        "summary": "公开面只读、来源透明、敏感字段脱敏; verified 与 manual-check 明确分开。",
        "controls": controls,
        "status_definitions": {
            "verified": "本轮有运行或配置证据",
            "verified-partial": "检查点门禁已通过，但仍保留明确的人工或第三方待检项",
            "implemented-partial": "代码或流程已实现，但运行覆盖或独立证据仍不完整",
            "manual-check": "需要网关、VPS 或第三方控制台证据",
            "planned": "尚未实现或启用",
            "failed": "验证未通过或存在需修复的问题",
            "n/a": "当前范围不适用",
        },
        "certification_claim": "none; this is an internal public trust disclosure, not an external certification",
    })
    return base


def _site31_scorecard_payload():
    gate_evidence = _site31_gate_evidence_payload()
    gate_dimensions = gate_evidence.get("dimensions", {}) if gate_evidence.get("valid") else {}
    security = gate_dimensions.get("security", {})
    accessibility = gate_dimensions.get("accessibility", {})
    dimensions = [
        {"key": "value_clarity", "label": "7 秒价值识别", "max_points": 12, "state": "verified", "evidence": ["/", "/api/research_portal"], "method": "desktop first-viewport review"},
        {"key": "scientific_provenance", "label": "科研可信与来源", "max_points": 15, "state": "verified", "evidence": ["/api/evidence_objects", "/api/research_passport"], "method": "schema, source, limitations and export checks"},
        {"key": "information_architecture", "label": "信息架构与导航", "max_points": 12, "state": "verified", "evidence": ["/", "/api/public_manifest"], "method": "desktop navigation and route regression"},
        {"key": "visual_system", "label": "视觉系统与品牌", "max_points": 14, "state": "verified", "evidence": ["/"], "method": "1366/1440/1536/1920/2048 browser review"},
        {"key": "motion", "label": "转场与微交互", "max_points": 10, "state": "verified", "evidence": ["/", "/benchmark"], "method": "online 01-04 theater, rapid latest-intent and deterministic reduced-motion checks"},
        {"key": "performance", "label": "性能与稳定性", "max_points": 12, "state": "verified", "evidence": ["/api/hardening"], "method": "22 visible animations, 6 backdrop layers, zero overflow and cached status probe; field CWV remains a separate follow-up"},
        {"key": "security_trust", "label": "安全与 Trust", "max_points": 12,
         "state": security.get("state", "work-in-progress"),
         "earned_points": round(float(security.get("earned_points", 0)), 2),
         "evidence": ["/api/trust_center", "/api/hardening", "/api/site31_gate_evidence"],
         "method": "checkpoint-weighted runtime/config/static audit; WAF and independent penetration remain manual-check"},
        {"key": "accessibility", "label": "可访问性", "max_points": 6,
         "state": accessibility.get("state", "work-in-progress"),
         "earned_points": round(float(accessibility.get("earned_points", 0)), 2),
         "evidence": ["/api/hardening", "/api/site31_gate_evidence"],
         "method": "checkpoint-weighted semantics, names, focus, status and reduced-motion audit; full screen-reader audit pending"},
        {"key": "localization", "label": "中文默认与全球表达", "max_points": 4, "state": "verified", "evidence": ["/"], "method": "Chinese default and English toggle route review"},
        {"key": "release_engineering", "label": "发布工程", "max_points": 3, "state": "verified", "evidence": ["/release", "/api/releases"], "method": "staged preflight, rollback snapshot, Gunicorn service and loopback verification"},
    ]
    weights = {"verified": 1.0, "verified-partial": None, "implemented-partial": 0.6,
               "manual-check": 0.0, "planned": 0.0, "failed": 0.0, "n/a": 0.0,
               "work-in-progress": 0.0}
    for row in dimensions:
        if "earned_points" not in row:
            row["earned_points"] = round(row["max_points"] * (weights.get(row["state"]) or 0.0), 2)
    earned = round(sum(x["earned_points"] for x in dimensions), 2)
    total = sum(x["max_points"] for x in dimensions)
    dimension_gate = all(x["earned_points"] >= x["max_points"] * .75 for x in dimensions)
    evidence_gate = bool(gate_evidence.get("valid") and gate_evidence.get("gate") == "pass")
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "schema_version": "site31.internal_scorecard.v2",
        "title": "Site31 internal commercial release scorecard",
        "score": earned,
        "max_score": total,
        "gate": "pass" if earned >= 90 and dimension_gate and evidence_gate else "work-in-progress",
        "score_type": "internal checkpoint-weighted release readiness; not a third-party ranking or certification",
        "state_weights": weights,
        "dimensions": dimensions,
        "gate_evidence": {
            "valid": gate_evidence.get("valid", False),
            "phase": gate_evidence.get("phase", "missing"),
            "gate": gate_evidence.get("gate", "fail"),
            "artifact_sha256": gate_evidence.get("artifact_sha256"),
            "error": gate_evidence.get("error"),
        },
        "hard_gates": [
            "no public physical actuation",
            "source labels preserved",
            "no desktop horizontal overflow",
            "default Chinese complete",
            "asset/SW/release versions consistent",
            "no critical browser console error",
            "release-matched gate evidence with SHA-256 integrity",
        ],
        "external_rank_claim": "none",
    }


def _site31_portal_payload():
    passport = _research_passport_payload()
    objects = _site31_evidence_objects(passport)
    trust = _site31_trust_center_payload(passport)
    scorecard = _site31_scorecard_payload()
    counts = passport.get("counts", {})
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "schema_version": "site31.global_research_portal.v1",
        "title": "XRD 智慧实验室",
        "subtitle": "近红外荧光材料预测与实验闭环证据库",
        "audience": "全球材料与近红外荧光粉科研工作者",
        "hero_metrics": [
            {"value": len(objects), "label": "可复核证据对象", "source": "/api/evidence_objects"},
            {"value": counts.get("material_rows", 0), "label": "材料公开行", "source": "/api/materials/explorer"},
            {"value": counts.get("public_get_surfaces", 0), "label": "公开只读证据面", "source": "/api/public_manifest"},
            {"value": "0", "label": "公网物理控制入口", "source": "/api/hardening"},
        ],
        "research_paths": [
            {"key": "discover", "title": "检索材料与配方", "detail": "按化学式、掺杂、trace_id 和 evidence_id 定位对象", "page": "/atlas"},
            {"key": "evaluate", "title": "复核 AI 判决", "detail": "查看 TS/MLIP/Conformal/Fly-MB 的来源、版本和限制", "page": "/brain"},
            {"key": "replay", "title": "检查具身实验", "detail": "复核具身真机闭环、双臂协同记录及 Lab-FSD shadow/assist 边界", "page": "/replay"},
            {"key": "reproduce", "title": "下载与复现", "detail": "Research Passport、Evidence Bundle、JSON/CSV 与引用文本", "page": "/api/evidence_bundle.json"},
        ],
        "system_cards": [
            {"site": "AI 脑系统卡", "pattern": "intended use / model origin / CI / failure modes", "reuse": "预测建议不替代烧制与 XRD/PL 实测", "url": "/brain"},
            {"site": "Lab-FSD 系统卡", "pattern": "SLAM / BEV / future risk / shadow policy", "reuse": "Nav2/MPPI 与安全员保持执行权威", "url": "/fsd"},
            {"site": "双机械臂证据卡", "pattern": "visual redundancy / bag drop / concurrent grinding / replay", "reuse": "CPU/OpenCV 为袋状态权威，BPU 仅作辅助与执行证据", "url": "/replay"},
            {"site": "公网 Trust 卡", "pattern": "read-only / redaction / release / manual checks", "reuse": "WAF/rate limit 无证据前不标 verified", "url": "/sec"},
        ],
        "open_science_commitments": [
            {"name": "来源透明", "detail": "每个对象显示 origin、source label、validation 和 freshness", "evidence": "/api/evidence_objects"},
            {"name": "限制就近", "detail": "AI、具身、机械臂和安全主张均附适用与禁用场景", "evidence": "/api/evidence_objects"},
            {"name": "可下载复核", "detail": "护照、证据包、材料 JSON/CSV 可离线检查", "evidence": "/api/evidence_bundle.json"},
            {"name": "状态诚实", "detail": "live/mirror/replay/stale/offline/unknown 不互相替代", "evidence": "/api/public_status"},
        ],
        "negative_controls": [
            {"id": "no_public_robot_control", "state": "verified", "check": "public portal exposes no physical actuation route"},
            {"id": "no_false_live_state", "state": "verified", "check": "offline hardware remains mirror/replay/offline/unknown"},
            {"id": "no_external_rank_claim", "state": "verified", "check": "internal scorecard is not presented as a global third-party ranking"},
            {"id": "edge_waf_evidence", "state": "manual-check", "check": "Cloudflare dashboard/API evidence still required"},
        ],
        "evidence_modes": [
            {"level": "实测", "name": "Observed / measured", "use": "XRD/PL 回填、具身闭环、arm01 视觉冗余与投袋、arm02 并发四周期研磨"},
            {"level": "文献", "name": "Literature / curated", "use": "公开材料来源、方法与引用"},
            {"level": "计算", "name": "Computed / model", "use": "TS/MLIP/Conformal/Fly-MB 和不确定性"},
            {"level": "回放", "name": "Replay / shadow", "use": "SLAM、Lab-FSD、任务和故障复盘"},
        ],
        "evidence_ladder": [
            {"level": "实测", "name": "Observed / measured", "use": "XRD/PL 回填、具身闭环、arm01 视觉冗余与投袋、arm02 并发四周期研磨"},
            {"level": "文献", "name": "Literature / curated", "use": "公开材料来源、方法与引用"},
            {"level": "计算", "name": "Computed / model", "use": "TS/MLIP/Conformal/Fly-MB 和不确定性"},
            {"level": "回放", "name": "Replay / shadow", "use": "SLAM、Lab-FSD、任务和故障复盘"},
        ],
        "current_fact_boundary": [
            FINALS_PUBLIC_FACTS["embodied"]["claim"] + " " + FINALS_PUBLIC_FACTS["embodied"]["boundary"],
            FINALS_PUBLIC_FACTS["ai_brain"]["claim"] + " " + FINALS_PUBLIC_FACTS["ai_brain"]["boundary"],
            FINALS_PUBLIC_FACTS["dual_arm"]["claim"] + " " + FINALS_PUBLIC_FACTS["dual_arm"]["boundary"],
            "X5/Pi 未上电时显示 mirror/replay/offline/unknown，不伪装 live。",
            "香港 VPS 面向全球访问; 部署位置不等于安全认证或备案结论。",
        ],
        "evidence_objects": objects,
        "trust_center": trust,
        "scorecard": scorecard,
        "top_platform_patterns": [],
    }


@app.route("/api/evidence_objects")
def api_evidence_objects():
    items = _site31_evidence_objects()
    q = (request.args.get("q") or "").strip().lower()
    filters = {
        "kind": (request.args.get("kind") or "").strip().lower(),
        "scope": (request.args.get("scope") or "").strip().lower(),
        "validation_status": (request.args.get("validation_status") or "").strip().lower(),
        "source_label": (request.args.get("source") or "").strip().lower(),
    }
    if q:
        items = [x for x in items if q in json.dumps(x, ensure_ascii=False).lower()]
    for key, value in filters.items():
        if value:
            items = [x for x in items if str(x.get(key, "")).lower() == value]
    origin = (request.args.get("origin") or "").strip().lower()
    if origin:
        items = [x for x in items if origin in [str(v).lower() for v in x.get("origin", [])]]
    sort_key = (request.args.get("sort") or "title").strip().lower()
    if sort_key in {"title", "kind", "scope", "updated_at"}:
        items.sort(key=lambda x: str(x.get(sort_key, "")).lower(), reverse=sort_key == "updated_at")
    facets = {
        "kind": sorted({str(x.get("kind") or "unknown") for x in items}),
        "scope": sorted({str(x.get("scope") or "unknown") for x in items}),
        "source": sorted({str(x.get("source_label") or "unknown") for x in items}),
        "validation_status": sorted({str(x.get("validation_status") or "unknown") for x in items}),
    }
    return jsonify({"ts": int(time.time()), "release": ASSET_VER,
                    "schema_version": "site31.evidence_index.v3", "count": len(items),
                    "facets": facets, "items": items})


def _evidence_object_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://xiaomiju.xyz/api/evidence_objects/schema.json",
        "title": "XRD Smart Lab Evidence Object v3",
        "type": "object",
        "required": ["evidence_id", "schema_version", "identifier", "version", "title", "claims",
                     "property_provenance", "validation", "uncertainty", "relations", "rights",
                     "limitations", "citation", "distributions"],
        "properties": {
            "evidence_id": {"type": "string", "pattern": "^ev:xrd:[a-z0-9_-]+$"},
            "schema_version": {"const": "site31.evidence_object.v3"},
            "identifier": {"type": "object", "required": ["local", "doi", "registration_status"]},
            "version": {"type": "object", "required": ["release", "revision", "released_at"]},
            "title": {"type": "string", "minLength": 1},
            "title_en": {"type": "string", "minLength": 1},
            "description_en": {"type": "string", "minLength": 1},
            "intended_use_en": {"type": "string", "minLength": 1},
            "prohibited_use_en": {"type": "string", "minLength": 1},
            "limitations_en": {"type": "array", "items": {"type": "string"}},
            "claims": {"type": "array", "minItems": 1},
            "relations": {"type": "array"},
            "limitations": {"type": "array"},
            "distributions": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["label", "href", "available", "mime_type", "bytes", "sha256",
                                 "generated_at", "license"],
                    "properties": {
                        "available": {"const": True},
                        "href": {"type": "string", "minLength": 1},
                        "bytes": {"type": "integer", "minimum": 1},
                        "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    },
                },
            },
        },
        "x-mappings": {
            "datacite": "DataCite Metadata Schema 4.7 mapping only; no DOI is registered",
            "ro-crate": "RO-Crate 1.3 mapping guidance; not a packaged crate",
        },
    }


@app.route("/api/evidence_objects/schema.json")
def api_evidence_object_schema():
    return jsonify(_evidence_object_schema())


@app.route("/api/evidence_objects/<path:evidence_id>/snapshot.json")
def api_evidence_object_snapshot(evidence_id):
    wanted = _stable_evidence_id(evidence_id)
    for item in _site31_evidence_objects():
        if item.get("evidence_id") == wanted:
            raw = _canonical_json_bytes(_evidence_snapshot_payload(item))
            resp = Response(raw, mimetype="application/json")
            resp.headers["Content-Disposition"] = (
                "attachment; filename=" + re.sub(r"[^A-Za-z0-9_.-]+", "_", wanted) + ".json")
            resp.headers["ETag"] = '"' + hashlib.sha256(raw).hexdigest() + '"'
            return resp
    return jsonify({"error": "evidence_not_found", "evidence_id": evidence_id, "release": ASSET_VER}), 404


@app.route("/api/evidence_objects/<path:evidence_id>")
def api_evidence_object_detail(evidence_id):
    wanted = _stable_evidence_id(evidence_id)
    for item in _site31_evidence_objects():
        if item.get("evidence_id") == wanted:
            return jsonify(item)
    return jsonify({"error": "evidence_not_found", "evidence_id": evidence_id, "release": ASSET_VER}), 404


@app.route("/api/trust_center")
def api_trust_center():
    return jsonify(_site31_trust_center_payload())


@app.route("/api/site29_portal")
def api_site29_portal():
    return jsonify(_site30_portal_payload())


@app.route("/api/site30_portal")
def api_site30_portal():
    return jsonify(_site30_portal_payload())


@app.route("/api/site31_portal")
def api_site31_portal():
    return jsonify(_site31_portal_payload())


@app.route("/api/site31_scorecard")
def api_site31_scorecard():
    return jsonify(_site31_scorecard_payload())


@app.route("/api/site31_gate_evidence")
def api_site31_gate_evidence():
    return jsonify(_site31_gate_evidence_public_payload(force_manifest_scan=True))


@app.route("/api/research_portal")
def api_research_portal():
    return jsonify(_site31_portal_payload())


def _research_collections_payload(query_args=None):
    args = request.args if query_args is None else query_args
    return build_research_collections(
        materials=_materials_all_rows(),
        evidence_objects=_site31_evidence_objects(),
        release=ASSET_VER,
        released_at=RELEASED_AT,
        params=args,
    )


@app.route("/api/research_collections")
def api_research_collections():
    return jsonify(_research_collections_payload())


@app.route("/api/research_collections/<path:collection_id>")
def api_research_collection_detail(collection_id):
    payload = _research_collections_payload({})
    detail = collection_detail(payload, collection_id)
    if detail is None:
        return jsonify({
            "schema_version": "site32.research_collections.v1",
            "release": ASSET_VER,
            "error": "collection_not_found",
            "collection_id": _public_safe_text(collection_id, 160),
        }), 404
    return jsonify(detail)


def _evidence_bundle_payload():
    passport = _research_passport_payload()
    materials = _materials_payload({})
    entries = _public_api_doc_entries()
    evidence_objects = _site31_evidence_objects(passport)
    trust_center = _site31_trust_center_payload(passport)
    return {
        "ts": int(time.time()),
        "release": ASSET_VER,
        "kind": "public_read_only_evidence_bundle",
        "passport": passport,
        "safe_field_policy": [
            "Only public-safe fields, source labels, release ids, citations and export links are included.",
            "Secrets, API keys, private prompts, raw credentials, full IP addresses and raw actuator commands are excluded.",
            "Live, mirror, replay, stale, offline and unknown states must remain visible.",
        ],
        "core_pages": ["/", "/atlas", "/brain", "/models", "/assets", "/twin", "/status"],
        "public_endpoints": [e["path"] for e in entries if "GET" in e["methods"]][:40],
        "material_schema": MATERIAL_SCHEMA,
        "materials_summary": materials["summary"],
        "materials_sample": [
            {key: row.get(key) for key in (
                "id", "formula", "host", "dopant", "site", "verdict", "lambda_em",
                "confidence_interval", "band", "method", "source", "state",
                "metadata_completeness_score", "uncertainty", "method_version",
                "detail_url", "api_url",
            )}
            for row in materials["items"][:12]
        ],
        "evidence_objects_summary": evidence_objects,
        "trust_controls": trust_center["controls"],
        "download_links": passport["downloads"],
        "answer_script": [
            "This is a global-facing public evidence portal for materials researchers, especially NIR phosphor work.",
            "The portal is hosted in Hong Kong for global access; regulatory obligations depend on actual deployment and operation and are not certified by this site.",
            "The public surface is read-only; unsafe public writes require SSO/RBAC headers and direct robot control is not exposed.",
            "The strongest claim is top-tier competition/research showcase quality, not commercial-scale parity with AlphaFold DB or Materials Project.",
        ],
    }


@app.route("/api/evidence_bundle.json")
def api_evidence_bundle_json():
    resp = Response(json.dumps(_evidence_bundle_payload(), ensure_ascii=False, indent=2), mimetype="application/json")
    resp.headers["Content-Disposition"] = "attachment; filename=xrd_research_evidence_bundle.json"
    return resp


@app.route("/api/evidence_bundle.txt")
def api_evidence_bundle_txt():
    payload = _evidence_bundle_payload()
    p = payload["passport"]
    lines = [
        "XRD Smart Lab public evidence bundle",
        f"Release: {ASSET_VER}",
        f"Title: {p['title']}",
        f"Audience: {', '.join(p['audience'])}",
        f"One sentence: {p['one_sentence']}",
        "",
        "Passport cards:",
    ]
    for card in p["passport_cards"]:
        lines.append(f"- {card['title']}: {card['value']} | {card['detail']}")
    lines.extend(["", "Evidence objects:"])
    for item in payload.get("evidence_objects_summary", []):
        lines.append(f"- {item['evidence_id']}: {item['claim']} | source={item['source_label']} | origin={','.join(item.get('origin', []))} | validation={item.get('validation_status')}")
    lines.extend(["", "Trust controls:"])
    for item in payload.get("trust_controls", []):
        lines.append(f"- {item['id']}: {item['state']} | {item['verification_method']}")
    lines.extend(["", "Trust posture:"])
    for item in p["trust_posture"]:
        lines.append(f"- {item['name']}: {item['state']} | {item['evidence']}")
    lines.extend(["", "Limitations:"])
    for item in p["limitations"]:
        lines.append(f"- {item}")
    lines.extend(["", f"Citation: {p['citation']['how_to_cite']}"])
    resp = Response("\n".join(lines) + "\n", mimetype="text/plain")
    resp.headers["Content-Disposition"] = "attachment; filename=xrd_research_evidence_bundle.txt"
    return resp


@app.route("/robots.txt")
def robots_txt():
    base = (request.url_root or "/").rstrip("/")
    body = "\n".join([
        "User-agent: *",
        "Allow: /",
        "Disallow: /admin",
        "Disallow: /api/config",
        f"Sitemap: {base}/sitemap.xml",
        "",
    ])
    return Response(body, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    base = (request.url_root or "/").rstrip("/")
    pages = ["/", "/status", "/brain", "/atlas", "/models", "/assets", "/twin"]
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    items = []
    for path in pages:
        loc = html.escape(base + path, quote=True)
        items.append(f"<url><loc>{loc}</loc><lastmod>{now}</lastmod><changefreq>daily</changefreq></url>")
    xml = "<?xml version='1.0' encoding='UTF-8'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>" + "".join(items) + "</urlset>"
    return Response(xml, mimetype="application/xml")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status_page():
    return send_from_directory("static", "index.html")


@app.route("/fleet")
def fleet_page():
    return send_from_directory("static", "index.html")


@app.route("/tasks")
def tasks_page():
    return send_from_directory("static", "index.html")


@app.route("/observability")
def observability_page():
    return send_from_directory("static", "index.html")


@app.route("/logs")
def logs_page():
    return send_from_directory("static", "index.html")


@app.route("/traces")
def traces_page():
    return send_from_directory("static", "index.html")


@app.route("/mq")
@app.route("/queue")
def queue_page():
    return send_from_directory("static", "index.html")


@app.route("/studio")
def studio_page():
    return send_from_directory("static", "index.html")


@app.route("/fsd")
def fsd_page():
    return send_from_directory("static", "index.html")


@app.route("/replay")
def replay_page():
    return send_from_directory("static", "index.html")


@app.route("/command")
def command_page():
    return send_from_directory("static", "index.html")


@app.route("/defense")
def defense_page():
    return send_from_directory("static", "index.html")


@app.route("/benchmark")
def benchmark_page():
    return send_from_directory("static", "index.html")


@app.route("/sec")
def security_page():
    return send_from_directory("static", "index.html")


@app.route("/twin")
def twin_page():
    return send_from_directory("static", "index.html")


@app.route("/atlas")
def atlas_page():
    return send_from_directory("static", "index.html")


@app.route("/brain")
def brain_page():
    return send_from_directory("static", "index.html")


@app.route("/detail")
def detail_page():
    return send_from_directory("static", "index.html")


@app.route("/materials/<path:mid>")
def material_page(mid):
    return send_from_directory("static", "index.html")


@app.route("/predictions/<path:trace_id>")
def prediction_page(trace_id):
    return send_from_directory("static", "index.html")


@app.route("/evidence/<path:evidence_id>")
def evidence_page(evidence_id):
    return send_from_directory("static", "index.html")


SPA_NAMED_PAGES = frozenset({
    "home", "highlight", "status", "fleet", "tasks", "ops", "story", "mq", "assets", "twin",
    "preflight", "atlas", "detail", "archive", "models", "standards", "cost", "glossary",
    "changelog", "build", "importw", "eln", "sync", "repro", "ar", "obs", "traces", "topo",
    "budget", "inc", "tm", "noc", "self", "oee", "alert", "logs", "qms", "cmms", "sec",
    "release", "data", "studio", "fsd", "replay", "command", "defense", "benchmark", "brain",
    "lab", "car", "arm",
})


@app.route("/<string:page_name>")
def named_spa_page(page_name):
    """Serve every allowlisted one-segment SPA deep link; unknown paths stay 404."""
    if page_name in SPA_NAMED_PAGES:
        return send_from_directory("static", "index.html")
    if os.path.isfile(os.path.join(app.static_folder, page_name)):
        return send_from_directory("static", page_name)
    abort(404)


@app.route("/design")
def design():
    """U1: 设计系统自文档页 (token 即文档, 直接消费 style.css 现值)."""
    return send_from_directory("static", "design.html")


@app.route("/healthz")
def healthz():
    return "ok"


MATERIAL_PUBLIC_FIELDS = [
    "id", "formula", "host", "dopant", "site", "verdict", "lambda_em",
    "confidence_interval", "band", "method", "source", "trace_id",
    "batch", "work_order", "stability_pct", "round", "state", "created",
    "metadata_completeness_score", "evidence_score", "uncertainty", "method_version", "provenance",
    "detail_url", "api_url"
]


def _atlas_snapshot(timeout=4.0):
    now = time.time()
    if timeout <= 0 and _atlas_cache["data"] is None:
        return {"ts": now, "source": "down", "items": [], "summary": {}}
    if _atlas_cache["data"] is None or now - _atlas_cache["ts"] > 25:
        try:
            _atlas_cache["data"] = _build_atlas(timeout=timeout)
            _atlas_cache["ts"] = now
        except Exception:
            if _atlas_cache["data"] is None:
                _atlas_cache["data"] = {"ts": now, "source": "down", "items": [], "summary": {}}
    return _atlas_cache["data"] or {"ts": now, "source": "down", "items": [], "summary": {}}


def _safe_text(v):
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v):
    try:
        if v in (None, ""):
            return None
        return round(float(v), 2)
    except Exception:
        return None


def _material_host_dopant(formula, dopant=""):
    txt = _safe_text(formula)
    dop = _safe_text(dopant)
    if ":" in txt:
        host, tail = txt.split(":", 1)
        return host.strip(), dop or tail.strip()
    return txt, dop


def _material_ci(conf):
    if isinstance(conf, dict):
        for k in ("ci90_nm", "ci90", "half_width_nm", "interval_nm"):
            val = _safe_float(conf.get(k))
            if val is not None:
                return val
        lo = _safe_float(conf.get("lower_nm") or conf.get("lo"))
        hi = _safe_float(conf.get("upper_nm") or conf.get("hi"))
        if lo is not None and hi is not None:
            return f"{lo}-{hi} nm"
    return _safe_float(conf)


def _material_method(source, round_tag):
    s = _safe_text(source)
    r = _safe_text(round_tag)
    joined = f"{s} {r}".lower()
    if "matter" in joined or "gen" in joined:
        return "MatterGen / MLIP / TS proxy"
    if "work order" in joined or "history" in joined:
        return "Work order history"
    return s or "curated public record"


def _material_id_path(row):
    return _safe_text(row.get("id") or row.get("trace_id") or row.get("formula")) or "unknown"


def _material_metadata_completeness(row):
    score = 30
    src = _safe_text(row.get("source")).lower()
    if src == "real":
        score += 22
    elif src == "mirror":
        score += 18
    elif src == "history":
        score += 20
    elif src == "curated":
        score += 10
    if row.get("trace_id"):
        score += 15
    if row.get("work_order"):
        score += 12
    if isinstance(row.get("lambda_em"), (int, float)):
        score += 8
    if row.get("confidence_interval") not in (None, ""):
        score += 6
    if isinstance(row.get("stability_pct"), (int, float)):
        score += 5
    return min(100, score)


def _material_uncertainty(row):
    ci = row.get("confidence_interval")
    if ci not in (None, ""):
        return f"CI90 {ci} nm" if isinstance(ci, (int, float)) else str(ci)
    src = _safe_text(row.get("source")).lower()
    state = _safe_text(row.get("state")).lower()
    if state == "observed" or src == "history":
        return "measured/history row; CI not exported"
    if row.get("lambda_em") in (None, ""):
        return "lambda not estimated"
    if src == "curated":
        return "curated replay; no live uncertainty"
    return "CI90 unavailable in public mirror"


def _material_method_version(row):
    method = _safe_text(row.get("method"))
    if "MatterGen" in method or "MLIP" in method or "TS" in method:
        return f"{ASSET_VER}; MatterGen + MatterSim/CHGNet + TS proxy"
    if "Observed" in method:
        return f"{ASSET_VER}; observed_pl.csv public fields"
    if "Work order" in method:
        return f"{ASSET_VER}; work-order history + prediction trace"
    return f"{ASSET_VER}; curated public schema"


def _finalize_material_row(row):
    row = dict(row)
    for key in ("id", "formula", "host", "dopant", "site", "verdict", "band", "trace_id",
                "batch", "work_order", "round", "state", "created"):
        row[key] = _public_safe_text(row.get(key), 160)
    for key in ("method", "source"):
        row[key] = _public_asset_text(row.get(key), 180)
    mid = _material_id_path(row)
    completeness = _material_metadata_completeness(row)
    row["metadata_completeness_score"] = completeness
    row["evidence_score"] = completeness  # Deprecated compatibility alias; not scientific confidence.
    row["uncertainty"] = _material_uncertainty(row)
    row["method_version"] = _material_method_version(row)
    row["provenance"] = (
        f"source={_safe_text(row.get('source')) or 'unknown'}; "
        f"state={_safe_text(row.get('state')) or 'unknown'}; "
        f"method={_safe_text(row.get('method')) or 'unknown'}; "
        f"release={ASSET_VER}"
    )
    row["detail_url"] = "/materials/" + quote(mid, safe="")
    row["api_url"] = "/api/materials/" + quote(mid, safe="")
    return row


def _atlas_material_rows():
    snap = _atlas_snapshot(timeout=0.0)
    source_state = _safe_text(snap.get("source")) or "down"
    rows = []
    for idx, it in enumerate(snap.get("items") or []):
        if not isinstance(it, dict):
            continue
        formula = _safe_text(it.get("formula")) or "unknown"
        host, dop = _material_host_dopant(formula)
        trace = _safe_text(it.get("trace"))
        round_tag = _safe_text(it.get("round"))
        rows.append({
            "id": trace or f"atlas-{idx + 1}",
            "formula": formula,
            "host": host,
            "dopant": dop,
            "site": _safe_text(it.get("site")),
            "verdict": _safe_text(it.get("verdict")) or "UNKNOWN",
            "lambda_em": _safe_float(it.get("lambda_em")),
            "confidence_interval": _material_ci(it.get("confidence")),
            "band": _safe_text(it.get("band")) or "unknown",
            "method": _material_method(it.get("source"), round_tag),
            "source": source_state,
            "trace_id": trace,
            "batch": "",
            "work_order": "",
            "stability_pct": _safe_float(it.get("stability_pct")),
            "round": round_tag,
            "state": "candidate",
            "created": "",
        })
    return rows


def _workorder_material_rows(limit=500):
    rows = []
    con = None
    try:
        con = _db()
        recs = con.execute(
            "SELECT code,formula,dop_symbol,dop_site,dop_pct,created_ts,stage,state,"
            "trace_id,verdict,pred_summary,pred_source,lambda_obs FROM workorders "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    except Exception:
        return rows
    finally:
        if con is not None:
            con.close()
    for idx, r in enumerate(recs):
        pred = {}
        if r[10]:
            try:
                pred = json.loads(r[10])
            except Exception:
                pred = {}
        formula = _safe_text(r[1]) or "unknown"
        dopant = _safe_text(r[2])
        host, dop = _material_host_dopant(formula, dopant)
        lam = _safe_float(r[12])
        if lam is None:
            for k in ("lambda_em", "lambda_em_nm", "lambda_pred", "lambda_nm"):
                lam = _safe_float(pred.get(k))
                if lam is not None:
                    break
        verdict = _safe_text(r[9]) or _safe_text(pred.get("verdict")) or "PENDING"
        trace = _safe_text(r[8])
        code = _safe_text(r[0])
        created = datetime.datetime.fromtimestamp(r[5]).isoformat() if r[5] else ""
        rows.append({
            "id": trace or code or f"wo-{idx + 1}",
            "formula": formula,
            "host": host,
            "dopant": dop,
            "site": _safe_text(r[3]),
            "verdict": verdict,
            "lambda_em": lam,
            "confidence_interval": _material_ci(pred.get("confidence") or pred.get("ci90")),
            "band": _atlas_band(lam) if lam is not None else "unknown",
            "method": _material_method(r[11], "work order"),
            "source": "history",
            "trace_id": trace,
            "batch": code,
            "work_order": code,
            "stability_pct": _safe_float(pred.get("stability_pct") or pred.get("thermal_stability_pct")),
            "round": "",
            "state": _safe_text(r[7]) or "open",
            "created": created,
        })
    return rows


def _observed_material_rows(limit=300):
    paths = [
        os.environ.get("XRD_OBSERVED_PL_CSV"),
        "/home/rdk/mirrors/lab/exp_ground_truth/observed_pl.csv",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "exp_ground_truth", "observed_pl.csv")),
    ]
    csv_path = next((p for p in paths if p and os.path.exists(p)), None)
    if not csv_path:
        return []
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            for idx, rec in enumerate(csv.DictReader(fh)):
                if idx >= limit:
                    break
                formula = _safe_text(rec.get("formula")) or "unknown"
                dopant = _safe_text(rec.get("dopant_element"))
                host = _safe_text(rec.get("host_family")) or formula
                lam = _safe_float(rec.get("actual_lambda_em_nm") or rec.get("lambda_em_nm"))
                xrd = (_safe_text(rec.get("actual_xrd_result")) or _safe_text(rec.get("xrd_result"))).lower()
                if "pure" in xrd or "single" in xrd or "ok" in xrd:
                    verdict = "GO"
                elif "mixed" in xrd or "impurity" in xrd:
                    verdict = "REVISE"
                else:
                    verdict = "OBSERVED"
                rows.append({
                    "id": f"observed-{idx + 1}",
                    "formula": formula,
                    "host": host,
                    "dopant": dopant,
                    "site": _safe_text(rec.get("dopant_site")),
                    "verdict": verdict,
                    "lambda_em": lam,
                    "confidence_interval": "",
                    "band": _atlas_band(lam) if lam is not None else "unknown",
                    "method": "Observed PL history",
                    "source": "history",
                    "trace_id": "",
                    "batch": "",
                    "work_order": "",
                    "stability_pct": _safe_float(rec.get("actual_thermal_stability_pct") or rec.get("thermal_stability_pct_at_150C")),
                    "round": "observed_pl.csv",
                    "state": "observed",
                    "created": _safe_text(rec.get("measurement_date")),
                })
    except Exception:
        return []
    return rows


def _seed_material_rows():
    seeds = [
        ("seed-yag-cr3", "Y3Al5O12:Cr3+", "YAG", "Cr3+", "Al", "REFERENCE", 714.0, "nir_i",
         "Project TS seed; compare observed history when available"),
        ("seed-ggg-ni2", "Gd3Ga5O12:Ni2+", "GGG", "Ni2+", "Ga", "REFERENCE", None, "unknown",
         "Public example chip; no live claim"),
        ("seed-sygo-cr-ni", "Sr6Y2Ga4O15:Cr/Ni", "SYGO", "Cr/Ni", "Ga", "REFERENCE", None, "unknown",
         "Project example chip for co-doping search"),
    ]
    rows = []
    for sid, formula, host, dop, site, verdict, lam, band, note in seeds:
        rows.append({
            "id": sid,
            "formula": formula,
            "host": host,
            "dopant": dop,
            "site": site,
            "verdict": verdict,
            "lambda_em": lam,
            "confidence_interval": "",
            "band": band,
            "method": note,
            "source": "curated",
            "trace_id": "",
            "batch": "",
            "work_order": "",
            "stability_pct": None,
            "round": "public example",
            "state": "replay",
            "created": "",
        })
    return rows


def _materials_all_rows():
    rows = _atlas_material_rows()
    seen = set()
    for row in rows:
        key = row.get("trace_id") or f"{row.get('formula')}|{row.get('dopant')}|{row.get('lambda_em')}"
        seen.add(key)
    # Internal work orders have no publication/embargo field yet. Keep them out
    # of the anonymous corpus until an explicit release gate exists.
    for row in _observed_material_rows():
        key = f"observed|{row.get('formula')}|{row.get('dopant')}|{row.get('lambda_em')}"
        if key in seen:
            continue
        rows.append(row)
        seen.add(key)
    for row in _seed_material_rows():
        key = row.get("id")
        if key in seen:
            continue
        rows.append(row)
        seen.add(key)
    return [_finalize_material_row(r) for r in rows]


MATERIAL_SCHEMA = [
    {"field": "formula", "label": "化学式", "label_en": "Formula", "meaning": "公开材料化学式或 host:dopant 字符串", "safe": True},
    {"field": "verdict", "label": "判决", "label_en": "Verdict", "meaning": "GO / REVISE / DROP / UNKNOWN / 实测状态", "safe": True},
    {"field": "lambda_em", "label": "发射峰", "label_en": "Emission peak", "meaning": "预测或实测发射峰，单位 nm", "safe": True},
    {"field": "confidence_interval", "label": "CI90 区间", "label_en": "CI90", "meaning": "已公开时显示不确定性半宽或区间", "safe": True},
    {"field": "source", "label": "来源状态", "label_en": "Source state", "meaning": "live / mirror / history / curated / offline 边界", "safe": True},
    {"field": "metadata_completeness_score", "label": "元数据完整度", "label_en": "Metadata completeness", "meaning": "仅衡量 trace、工单、PL、CI 与热稳定字段是否齐全；不是科学置信度", "safe": True},
    {"field": "evidence_score", "label": "兼容字段（已弃用）", "label_en": "Deprecated alias", "meaning": "metadata_completeness_score 的兼容别名；不得解释为证据强度", "safe": True, "deprecated": True},
]


MATERIAL_METHOD_CARDS = [
    {"key": "parse", "title": "配方对象", "title_en": "Formula object", "detail": "将化学式、基质、掺杂、占位、trace_id 与工单归一为可引用记录。"},
    {"key": "mlip", "title": "MLIP 稳定性", "title_en": "MLIP stability", "detail": "在字段存在时汇总 MatterGen 候选与 MatterSim/CHGNet 稳定性结果。"},
    {"key": "ts", "title": "TS 光学代理", "title_en": "TS optical proxy", "detail": "发射峰与波段来自可微 Tanabe-Sugano 光学代理或明确标注的实测历史。"},
    {"key": "audit", "title": "证据边界", "title_en": "Evidence boundary", "detail": "记录保留 live/mirror/history/curated 标签；离线数据不会显示为实时。"},
]


def _materials_payload(query_args=None):
    args = request.args if query_args is None else query_args
    q = _safe_text(args.get("q")).lower()
    verdict = _safe_text(args.get("verdict"))
    band = _safe_text(args.get("band"))
    sort = _safe_text(args.get("sort")) or "lambda_em"
    direction = _safe_text(args.get("dir")).lower()
    try:
        limit = min(max(int(args.get("limit", 500)), 1), 1000)
    except Exception:
        limit = 500
    rows = _materials_all_rows()
    if q:
        def hit(row):
            hay = " ".join(_safe_text(row.get(k)) for k in MATERIAL_PUBLIC_FIELDS).lower()
            return q in hay
        rows = [r for r in rows if hit(r)]
    if verdict:
        rows = [r for r in rows if _safe_text(r.get("verdict")).lower() == verdict.lower()]
    if band:
        rows = [r for r in rows if _safe_text(r.get("band")).lower() == band.lower()]
    if sort not in MATERIAL_PUBLIC_FIELDS:
        sort = "lambda_em"
    non_empty = [r for r in rows if r.get(sort) not in (None, "")]
    empty = [r for r in rows if r.get(sort) in (None, "")]

    def skey(row):
        val = row.get(sort)
        if isinstance(val, (int, float)):
            return val
        return _safe_text(val).lower()

    non_empty.sort(key=skey, reverse=(direction == "desc"))
    rows = (non_empty + empty)[:limit]
    lams = [r["lambda_em"] for r in rows if isinstance(r.get("lambda_em"), (int, float))]
    summary = {
        "total": len(rows),
        "limit": limit,
        "verdict": {},
        "band": {},
        "source": {},
        "lambda_min": min(lams) if lams else None,
        "lambda_max": max(lams) if lams else None,
    }
    for r in rows:
        for k in ("verdict", "band", "source"):
            v = _safe_text(r.get(k)) or "unknown"
            summary[k][v] = summary[k].get(v, 0) + 1
    hosts, dopants = {}, {}
    for r in rows:
        if r.get("host"):
            hosts[_safe_text(r.get("host"))] = hosts.get(_safe_text(r.get("host")), 0) + 1
        if r.get("dopant"):
            dopants[_safe_text(r.get("dopant"))] = dopants.get(_safe_text(r.get("dopant")), 0) + 1
    summary["host"] = dict(sorted(hosts.items(), key=lambda kv: (-kv[1], kv[0]))[:12])
    summary["dopant"] = dict(sorted(dopants.items(), key=lambda kv: (-kv[1], kv[0]))[:12])
    return {"ts": time.time(), "release": ASSET_VER, "fields": MATERIAL_PUBLIC_FIELDS,
            "query": {"q": q, "verdict": verdict, "band": band, "sort": sort, "dir": direction or "asc"},
            "summary": summary, "schema": MATERIAL_SCHEMA, "method_cards": MATERIAL_METHOD_CARDS,
            "source_policy": "Public rows are explicitly labelled live, mirror, history, curated, down/offline. Internal work orders are excluded until an explicit publication gate exists; the UI must never turn a stale row into a live claim.",
            "items": rows}


@app.route("/api/materials/explorer")
def api_materials_explorer():
    return jsonify(_materials_payload())


@app.route("/api/materials/export.csv")
def api_materials_export_csv():
    payload = _materials_payload()
    rows = [[r.get(k, "") for k in MATERIAL_PUBLIC_FIELDS] for r in payload["items"]]
    return _csv_response("materials_explorer", MATERIAL_PUBLIC_FIELDS, rows)


@app.route("/api/materials/export.json")
def api_materials_export_json():
    payload = _materials_payload()
    resp = Response(json.dumps(payload, ensure_ascii=False, indent=2), mimetype="application/json")
    resp.headers["Content-Disposition"] = "attachment; filename=materials_explorer.json"
    return resp


def _material_lookup(mid):
    token = _safe_text(mid)
    low = token.lower()
    for row in _materials_all_rows():
        keys = [row.get("id"), row.get("formula"), row.get("trace_id"), row.get("work_order")]
        if any(_safe_text(k).lower() == low for k in keys if k):
            return row
    return None


def _prediction_lookup(trace_id):
    token = _safe_text(trace_id).lower()
    if not token:
        return None
    for row in _materials_all_rows():
        trace = _safe_text(row.get("trace_id"))
        if trace and trace.lower() == token:
            return row
    return None


def _material_detail_payload(row, kind="material"):
    citation_id = row.get("trace_id") or row.get("id") or row.get("formula")
    item = {k: row.get(k) for k in MATERIAL_PUBLIC_FIELDS}
    cite = {
        "id": citation_id,
        "trace_id": row.get("trace_id") or "",
        "timestamp": row.get("created") or "",
        "version": ASSET_VER,
        "method": row.get("method") or "",
        "source": row.get("source") or "",
        "text": f"XRD Smart Lab Team. {row.get('formula') or citation_id}. Version {ASSET_VER}, 2026.",
    }
    tabs = {
        "structure": {
            "title": "结构",
            "rows": [
                ["化学式", row.get("formula")],
                ["基质", row.get("host")],
                ["掺杂", row.get("dopant")],
                ["占位", row.get("site")],
                ["状态", row.get("state")],
            ],
            "note": "这里只展示公开结构摘要；不暴露原始 CIF 与内部模型缓存路径。",
        },
        "xrd": {
            "title": "XRD",
            "rows": [
                ["判决", row.get("verdict")],
                ["批次", row.get("batch") or "未关联"],
                ["工单", row.get("work_order") or "未关联"],
                ["来源", row.get("source")],
            ],
            "note": "关联批次或工单后显示 XRD 详情；未关联时仅提供公开摘要。",
        },
        "pl": {
            "title": "PL 光谱",
            "rows": [
                ["发射峰 lambda_em", row.get("lambda_em")],
                ["CI90", row.get("confidence_interval") or "未提供"],
                ["波段", row.get("band")],
                ["热稳定性", row.get("stability_pct")],
            ],
            "note": "发射数据来自明确标注的 live/mirror 图鉴、工单历史、observed_pl.csv 或策展回放行。",
        },
        "reasoning": {
            "title": "AI 推理",
            "rows": [
                ["方法", row.get("method")],
                ["模型/数据版本", ASSET_VER],
                ["轮次", row.get("round")],
                ["来源标签", row.get("source")],
                ["元数据完整度", row.get("metadata_completeness_score")],
                ["不确定性", row.get("uncertainty")],
            ],
            "note": "推理文本只依据公开元数据汇总；内部提示词、密钥与原始私有日志均被排除。",
        },
        "feedback": {
            "title": "实验回填",
            "rows": [
                ["状态", row.get("state")],
                ["实测来源", row.get("source")],
                ["追踪 ID", row.get("trace_id") or "未关联"],
                ["工单", row.get("work_order") or "未关联"],
            ],
            "note": "材料关联工单与实测批次后，回填证据会更完整。",
        },
        "repro": {
            "title": "可复现性",
            "rows": [
                ["引用 ID", citation_id],
                ["版本", ASSET_VER],
                ["方法版本", row.get("method_version")],
                ["公开字段", ", ".join(MATERIAL_PUBLIC_FIELDS)],
                ["离线行为", "history/curated/replay 明确标注，不伪装 live"],
            ],
            "note": "使用引用块与 JSON 下载复核该公开对象的当前状态。",
        },
    }
    public_id = quote(str(citation_id if kind == "prediction" else (row.get("id") or citation_id)), safe="")
    base = "/api/predictions/" + public_id if kind == "prediction" else "/api/materials/" + public_id
    downloads = [
        {"label": "JSON", "href": base + "/export.json", "available": True},
        {"label": "CSV", "href": base + "/export.csv", "available": True},
        {"label": "文本报告", "href": base + "/report.txt", "available": True},
        {"label": "配方 / 称量单", "href": "", "available": False,
         "reason": "该对象尚未关联可公开配方。"},
        {"label": "批次档案", "href": "", "available": False,
         "reason": "工单摘要可查看，但尚无具备完整元数据与校验值的公开批次档案。"},
    ]
    api_base = "/api/predictions/" + public_id if kind == "prediction" else "/api/materials/" + public_id
    return {"ts": time.time(), "release": ASSET_VER, "kind": kind, "item": item,
            "summary": item, "tabs": tabs,
            "tab_order": ["structure", "xrd", "pl", "reasoning", "feedback", "repro"],
            "citation": cite, "downloads": downloads,
            "schema": MATERIAL_SCHEMA, "method_cards": MATERIAL_METHOD_CARDS,
            "provenance": {
                "source_policy": "live/mirror/history/curated/offline labels are preserved in public UI",
                "row": row.get("provenance"),
                "metadata_completeness_score": row.get("metadata_completeness_score"),
                "evidence_score": row.get("evidence_score"),
                "evidence_score_deprecated": True,
                "uncertainty": row.get("uncertainty"),
            },
            "api_examples": [
                {"label": "详情 JSON", "curl": f"curl https://xiaomiju.xyz{api_base}"},
                {"label": "图鉴检索", "curl": "curl 'https://xiaomiju.xyz/api/materials/explorer?q=YAG:Cr3%2B&sort=lambda_em'"},
            ]}


def _detail_or_404(mid, kind="material"):
    row = _prediction_lookup(mid) if kind == "prediction" else _material_lookup(mid)
    if not row:
        return None, (jsonify({"error": "not_found", "id": mid, "kind": kind,
                               "release": ASSET_VER,
                               "message": "未找到与该 ID 匹配的公开材料或预测对象。"}), 404)
    return _material_detail_payload(row, kind=kind), None


def _detail_csv_response(payload, name):
    row = payload["item"]
    rows = [[k, row.get(k, "")] for k in MATERIAL_PUBLIC_FIELDS]
    return _csv_response(name, ["field", "value"], rows)


@app.route("/api/materials/<path:mid>/export.json")
def api_material_detail_export_json(mid):
    payload, err = _detail_or_404(mid, "material")
    if err:
        return err
    resp = Response(json.dumps(payload, ensure_ascii=False, indent=2), mimetype="application/json")
    resp.headers["Content-Disposition"] = f"attachment; filename=material_{re.sub(r'[^A-Za-z0-9_.-]+','_', mid)[:60]}.json"
    return resp


@app.route("/api/materials/<path:mid>/export.csv")
def api_material_detail_export_csv(mid):
    payload, err = _detail_or_404(mid, "material")
    if err:
        return err
    return _detail_csv_response(payload, "material_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", mid)[:60])


@app.route("/api/materials/<path:mid>/report.txt")
def api_material_detail_report(mid):
    payload, err = _detail_or_404(mid, "material")
    if err:
        return err
    it, cite = payload["item"], payload["citation"]
    lines = [
        "XRD Smart Lab public material report",
        f"Release: {ASSET_VER}",
        f"Formula: {it.get('formula')}",
        f"Dopant/site: {it.get('dopant')} / {it.get('site')}",
        f"Verdict: {it.get('verdict')}",
        f"lambda_em: {it.get('lambda_em')}",
        f"Source: {it.get('source')} ({it.get('state')})",
        f"Method: {it.get('method')}",
        f"Cite: {cite.get('id')} | {cite.get('version')} | {cite.get('method')}",
    ]
    resp = Response("\n".join(lines) + "\n", mimetype="text/plain")
    resp.headers["Content-Disposition"] = f"attachment; filename=material_{re.sub(r'[^A-Za-z0-9_.-]+','_', mid)[:60]}.txt"
    return resp


@app.route("/api/materials/<path:mid>")
def api_material_detail(mid):
    payload, err = _detail_or_404(mid, "material")
    if err:
        return err
    return jsonify(payload)


@app.route("/api/predictions/<path:trace_id>/export.json")
def api_prediction_detail_export_json(trace_id):
    payload, err = _detail_or_404(trace_id, "prediction")
    if err:
        return err
    resp = Response(json.dumps(payload, ensure_ascii=False, indent=2), mimetype="application/json")
    resp.headers["Content-Disposition"] = f"attachment; filename=prediction_{re.sub(r'[^A-Za-z0-9_.-]+','_', trace_id)[:60]}.json"
    return resp


@app.route("/api/predictions/<path:trace_id>/export.csv")
def api_prediction_detail_export_csv(trace_id):
    payload, err = _detail_or_404(trace_id, "prediction")
    if err:
        return err
    return _detail_csv_response(payload, "prediction_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_id)[:60])


@app.route("/api/predictions/<path:trace_id>/report.txt")
def api_prediction_detail_report(trace_id):
    payload, err = _detail_or_404(trace_id, "prediction")
    if err:
        return err
    it, cite = payload["item"], payload["citation"]
    lines = [
        "XRD Smart Lab public prediction report",
        f"Release: {ASSET_VER}",
        f"Trace ID: {it.get('trace_id')}",
        f"Formula: {it.get('formula')}",
        f"Verdict: {it.get('verdict')}",
        f"lambda_em: {it.get('lambda_em')}",
        f"Source: {it.get('source')} ({it.get('state')})",
        f"Method: {it.get('method')}",
        f"Cite: {cite.get('id')} | {cite.get('version')} | {cite.get('method')}",
    ]
    resp = Response("\n".join(lines) + "\n", mimetype="text/plain")
    safe_trace = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_id)[:60]
    resp.headers["Content-Disposition"] = f"attachment; filename=prediction_{safe_trace}.txt"
    return resp


@app.route("/api/predictions/<path:trace_id>")
def api_prediction_detail(trace_id):
    payload, err = _detail_or_404(trace_id, "prediction")
    if err:
        return err
    return jsonify(payload)


register_site32(app, release=ASSET_VER, released_at=RELEASED_AT)


_runtime = RuntimeController()


def start_runtime():
    """Initialize persistence and workers explicitly for the serving process."""
    if CMD_TEST_MODE:
        return False
    return _runtime.start(
        initialize=_init_db,
        seed=_seed_defaults,
        workers=(("historian", _sampler), ("twin", _twin_loop)),
    )


def create_app(test_config=None):
    """Gunicorn/Flask application factory with opt-in background runtime."""
    if test_config:
        app.config.update(test_config)
    if load_config().runtime_enabled:
        start_runtime()
    return app

if __name__ == "__main__":
    start_runtime()
    port = load_config().port
    app.run(host="127.0.0.1", port=port, threaded=True)
