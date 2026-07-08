"""interlock — 双臂防互撞软件互锁 (第 4 期 #3, 2026-06-12).

纯 numpy-free 实现 (Pi 4B 2GB 上跑, math 就够): myCobot 280 正运动学算 6 关节
原点 → 相邻原点连成 5 段 capsule (胶囊体), 双臂两两段间最小距离 < 阈值 = 互锁.

两种用法:
  1. 实时监测: monitor_tick(j1, j2) 由 app.py 的 socket 流顺手调, 状态进 /api/interlock
  2. ghost 预演: check_move(arm, target_angles) 在 move/replay 下发**前**用目标角
     做 FK 对照另一臂当前角 — 预测撞就拒绝 (软件互锁, 不依赖 UI)

DH 参数 (标准 DH, 单位 mm/rad):
  Source: Elephant Robotics myCobot 280 URDF (mycobot_ros) / GitBook DH table.
  d1=131.22, a2=-110.4, a3=-96, d4=63.4, d5=75.05, d6=45.6
  ⚠ theta 偏置符号按官方表抄录; 真机上电后用已知姿态 (HOME [0,8,-127,40,0,45])
  验证末端 xyz 再收紧阈值 — 在那之前 CLEARANCE_MM 取保守 60mm.

双臂基座相对位姿: 工位 fixture 未定稿, 先配置文件 ~/interlock_config.json
  {"base_dx_mm": 400, "base_dy_mm": 0, "base_yaw_deg": 180}  (面对面 400mm 默认)
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

CONFIG_PATH = Path.home() / "interlock_config.json"

# ---- myCobot 280 标准 DH (mm / rad) ----
# Source: Elephant Robotics myCobot 280 URDF (mycobot_ros repo), GitBook DH table
_DH = [
    # (alpha,        a,      d,      theta_offset)
    (math.pi / 2,    0.0,    131.22, 0.0),
    (0.0,           -110.4,  0.0,   -math.pi / 2),
    (0.0,            -96.0,  0.0,    0.0),
    (math.pi / 2,    0.0,    63.4,  -math.pi / 2),
    (-math.pi / 2,   0.0,    75.05,  math.pi / 2),
    (0.0,            0.0,    45.6,   0.0),
]

LINK_RADIUS_MM = 40.0      # 链节胶囊半径 (含外壳 + 线缆余量, 工程保守值)
CLEARANCE_MM = 60.0        # 互锁阈值: 任意两 capsule 表面距 < 60mm 触发
WARN_MM = 120.0            # 预警黄区

_cfg = {"base_dx_mm": 400.0, "base_dy_mm": 0.0, "base_yaw_deg": 180.0, "enabled": True}
_status = {"min_dist_mm": None, "level": "unknown", "pair": None, "ts": 0.0,
           "blocks": 0, "last_block": None}


def load_config() -> dict:
    global _cfg
    try:
        if CONFIG_PATH.exists():
            _cfg.update(json.load(CONFIG_PATH.open(encoding="utf-8")))
    except Exception:
        pass
    return dict(_cfg)


def save_config(updates: dict) -> dict:
    for k in ("base_dx_mm", "base_dy_mm", "base_yaw_deg"):
        if k in updates:
            _cfg[k] = float(updates[k])
    if "enabled" in updates:
        _cfg["enabled"] = bool(updates["enabled"])
    try:
        CONFIG_PATH.write_text(json.dumps(_cfg), encoding="utf-8")
    except OSError:
        pass
    return dict(_cfg)


load_config()


# ---------------------------------------------------------------- FK
def _dh_mat(alpha: float, a: float, d: float, theta: float):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return (
        (ct, -st * ca,  st * sa, a * ct),
        (st,  ct * ca, -ct * sa, a * st),
        (0.0,      sa,       ca,      d),
    )  # 3x4 (省略恒为 0001 的第 4 行)


def _mat_mul(m1, m2):
    out = []
    for r in range(3):
        row = []
        for c in range(4):
            v = sum(m1[r][k] * m2[k][c] for k in range(3))
            if c == 3:
                v += m1[r][3]
            row.append(v)
        out.append(tuple(row))
    return tuple(out)


def fk_points(angles_deg: list[float], base: tuple[float, float, float] = (0, 0, 0),
              base_yaw_deg: float = 0.0) -> list[tuple[float, float, float]]:
    """6 关节角 (deg) → 7 个原点坐标 (基座 + 6 关节末) mm, 世界系."""
    yaw = math.radians(base_yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    m = (
        (cy, -sy, 0.0, base[0]),
        (sy,  cy, 0.0, base[1]),
        (0.0, 0.0, 1.0, base[2]),
    )
    pts = [(m[0][3], m[1][3], m[2][3])]
    for i, (alpha, a, d, off) in enumerate(_DH):
        theta = math.radians(angles_deg[i]) + off
        m = _mat_mul(m, _dh_mat(alpha, a, d, theta))
        pts.append((m[0][3], m[1][3], m[2][3]))
    return pts


# ------------------------------------------------- segment-segment distance
def _seg_dist(p1, q1, p2, q2) -> float:
    """两线段最小距离 (Ericson, Real-Time Collision Detection §5.1.9 闭式解)."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    d1, d2, r = sub(q1, p1), sub(q2, p2), sub(p1, p2)
    a, e, f = dot(d1, d1), dot(d2, d2), dot(d2, r)
    EPS = 1e-9
    if a <= EPS and e <= EPS:
        return math.sqrt(dot(r, r))
    if a <= EPS:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = dot(d1, r)
        if e <= EPS:
            t, s = 0.0, max(0.0, min(1.0, -c / a))
        else:
            b = dot(d1, d2)
            den = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / den)) if den > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, max(0.0, min(1.0, (b - c) / a))
    cp1 = (p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s)
    cp2 = (p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t)
    v = sub(cp1, cp2)
    return math.sqrt(dot(v, v))


def min_capsule_distance(angles1: list[float], angles2: list[float]) -> tuple[float, tuple[int, int]]:
    """双臂 capsule 间最小**表面**距离 mm + 最近段对 (i, j). arm2 在配置的基座位姿."""
    pts1 = fk_points(angles1)
    pts2 = fk_points(angles2, base=(_cfg["base_dx_mm"], _cfg["base_dy_mm"], 0.0),
                     base_yaw_deg=_cfg["base_yaw_deg"])
    best, pair = float("inf"), (0, 0)
    # 跳过段 0 (基座→J1 竖直柱, 两臂基座固定不可能互撞, 省 1/6 计算)
    for i in range(1, len(pts1) - 1):
        for j in range(1, len(pts2) - 1):
            d = _seg_dist(pts1[i], pts1[i + 1], pts2[j], pts2[j + 1])
            if d < best:
                best, pair = d, (i, j)
    return best - 2 * LINK_RADIUS_MM, pair


# ---------------------------------------------------------------- 互锁 API
def monitor_tick(angles1: list[float] | None, angles2: list[float] | None) -> dict:
    """app.py 周期调 (跟 joint 流同节拍). 返回并缓存状态."""
    if not _cfg.get("enabled", True) or not angles1 or not angles2:
        _status.update(min_dist_mm=None, level="off" if not _cfg.get("enabled") else "unknown",
                       pair=None, ts=time.time())
        return dict(_status)
    d, pair = min_capsule_distance(angles1, angles2)
    level = "danger" if d < CLEARANCE_MM else "warn" if d < WARN_MM else "ok"
    _status.update(min_dist_mm=round(d, 1), level=level, pair=pair, ts=time.time())
    return dict(_status)


def check_move(target_angles: list[float], other_angles: list[float] | None,
               moving_is_arm1: bool = True) -> dict:
    """ghost 预演: 目标角 FK vs 另一臂当前角. ok=False 时调用方应拒绝下发."""
    if not _cfg.get("enabled", True):
        return {"ok": True, "level": "off", "min_dist_mm": None}
    if not other_angles:
        return {"ok": True, "level": "unknown", "min_dist_mm": None,
                "note": "另一臂无数据, 互锁放行 (单臂模式)"}
    if moving_is_arm1:
        d, pair = min_capsule_distance(target_angles, other_angles)
    else:
        d, pair = min_capsule_distance(other_angles, target_angles)
    ok = d >= CLEARANCE_MM
    if not ok:
        _status["blocks"] = _status.get("blocks", 0) + 1
        _status["last_block"] = {"t": time.time(), "min_dist_mm": round(d, 1), "pair": pair}
    return {"ok": ok, "level": "danger" if not ok else "warn" if d < WARN_MM else "ok",
            "min_dist_mm": round(d, 1), "pair": pair,
            "clearance_mm": CLEARANCE_MM}


def status() -> dict:
    return {**_status, "config": dict(_cfg),
            "clearance_mm": CLEARANCE_MM, "warn_mm": WARN_MM,
            "link_radius_mm": LINK_RADIUS_MM}


def ghost_points(angles1: list[float], angles2: list[float]) -> dict:
    """前端 2D 顶视图 / 3D ghost 用: 双臂 7 点世界坐标."""
    return {
        "arm01": fk_points(angles1),
        "arm02": fk_points(angles2, base=(_cfg["base_dx_mm"], _cfg["base_dy_mm"], 0.0),
                           base_yaw_deg=_cfg["base_yaw_deg"]),
    }
